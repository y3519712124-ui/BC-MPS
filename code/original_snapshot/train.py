"""Validation-only development runner for boundary-aware label reliability.

This is a method candidate, not a verified novel or superior algorithm.
Test metrics are intentionally not computed during development.
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from protocol import Scene, Patches, digest


class Classifier(nn.Module):
    def __init__(self,bands,classes,width=32,early_fusion=False):
        super().__init__()
        self.early_fusion=early_fusion
        def encoder(c):
            return nn.Sequential(nn.Conv2d(c,width,3,padding=1),nn.GroupNorm(4,width),nn.GELU(),
                                 nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(4,width),nn.GELU())
        self.h=encoder(bands+1) if early_fusion else encoder(bands)
        self.d=None if early_fusion else encoder(1)
        self.head_h=nn.Linear(width,classes); self.head_d=nn.Linear(width,classes)
        self.fused=nn.Linear(2*width,classes)

    def forward(self,h,d):
        # Centre prediction keeps spectral/elevation signals at the queried pixel.
        if self.early_fusion:
            a=self.h(torch.cat([h,d],1)); c=a.shape[-1]//2
            a=a[:,:,c,c]; b=torch.zeros_like(a)
        else:
            a=self.h(h); b=self.d(d); c=a.shape[-1]//2
            a=a[:,:,c,c]; b=b[:,:,c,c]
        return self.fused(torch.cat([a,b],1)),self.head_h(a),self.head_d(b)


class PublishedReimplementation(nn.Module):
    """Lightweight structure-matched reimplementation of a published residual
    HSI--LiDAR fusion family.  It is deliberately labelled as a paper-structure
    reimplementation because the original authors' training code is unavailable
    in this environment; it is not claimed to be an official-code reproduction.
    """
    def __init__(self,bands,classes,width=32):
        super().__init__()
        def branch(c):
            stem=nn.Sequential(nn.Conv2d(c,width,3,padding=1),nn.BatchNorm2d(width),nn.ReLU())
            block=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.BatchNorm2d(width),nn.ReLU(),
                                nn.Conv2d(width,width,3,padding=1),nn.BatchNorm2d(width))
            return stem,block
        self.hs,self.hb=branch(bands); self.ds,self.db=branch(1)
        self.fuse=nn.Sequential(nn.Conv2d(2*width,width,1),nn.BatchNorm2d(width),nn.ReLU())
        self.head=nn.Linear(width,classes)
    def forward(self,h,d):
        ha=self.hs(h); hb=F.relu(ha+self.hb(ha))
        da=self.ds(d); db=F.relu(da+self.db(da))
        f=self.fuse(torch.cat([hb,db],1)); c=f.shape[-1]//2
        z=f[:,:,c,c]; logits=self.head(z)
        # Keep the same three-output contract as the main model for evaluation.
        return logits, logits, logits


def boundary_proxy(h,d):
    """Patch-edge signal from inputs, without labels or injected noise flags."""
    # RMS spectral differences preserve bandwise edges before reduction.
    def gradient(x):
        gx=(x[:,:,:,1:]-x[:,:,:,:-1]).square().mean(1).sqrt()
        gy=(x[:,:,1:,:]-x[:,:,:-1,:]).square().mean(1).sqrt()
        return .5*(gx.mean((1,2))+gy.mean((1,2)))
    return torch.tanh(.5*(gradient(h)+gradient(d)))


@torch.no_grad()
def label_reliability(predictions,y,boundary,use_boundary=True):
    pf,ph,pd=[x.softmax(1) for x in predictions]
    index=y[:,None]
    support=(ph.gather(1,index).squeeze(1)*pd.gather(1,index).squeeze(1)).sqrt()
    # At high gradients, modality disagreement may reflect legitimate mixed
    # evidence. Reduce dependence on strict cross-modal agreement there.
    if use_boundary:
        # Boundary pixels have a higher mixed-pixel risk; reduce their label
        # weight rather than replacing them with a teacher pseudo-label.
        support=support*(1-0.5*boundary)
    return support.clamp(.05,.95),pf


@torch.no_grad()
def evaluate(model,loader,scene):
    model.eval(); conf=torch.zeros(scene.classes,scene.classes,dtype=torch.int64)
    boundary_correct=boundary_n=0
    device=next(model.parameters()).device
    for h,d,y,i in loader:
        logits,_,_=model(h.to(device),d.to(device)); pred=logits.argmax(1).cpu()
        conf+=torch.bincount(y*scene.classes+pred,minlength=scene.classes**2).reshape(scene.classes,scene.classes)
        coords=loader.dataset.coords[i.numpy()]
        flags=torch.from_numpy(scene.boundary[tuple(coords.T)])
        boundary_correct+=int(((pred==y)&flags).sum()); boundary_n+=int(flags.sum())
    n=int(conf.sum()); counts=conf.sum(1); present=counts>0
    accuracy=conf.diag().double()/counts.clamp_min(1)
    po=float(conf.diag().sum()/n); pe=float((conf.sum(0).double()*counts).sum()/n**2)
    return {'oa':po*100,'aa_present_classes':float(accuracy[present].mean()*100),
            'kappa':100*(po-pe)/(1-pe) if pe<1 else None,
            'observed_boundary_oa':100*boundary_correct/boundary_n if boundary_n else None,
            'boundary_pixels':boundary_n,'class_counts':counts.tolist(),
            'per_class_recall':[float(accuracy[c]*100) if present[c] else None for c in range(scene.classes)],
            'confusion':conf.tolist(),'pixels':n}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--method',choices=['ce','cross_modal','boundary_reliability','gce_boundary','early_fusion','published_reimplementation'],default='ce')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--noise',type=float,default=.2);p.add_argument('--noise-model',choices=['uniform','boundary'],default='uniform')
    p.add_argument('--block',type=int,default=64);p.add_argument('--patch',type=int,default=11)
    p.add_argument('--split-seed',type=int,default=20261005);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--gce-q',type=float,default=.7)
    p.add_argument('--device',choices=['cpu'],default='cpu')
    p.add_argument('--ablation',choices=['none','hsi_only','lidar_only'],default='none')
    a=p.parse_args()
    if not 0 < a.gce_q <= 1:
        p.error('--gce-q must be in (0,1]')
    a.out.mkdir(parents=True,exist_ok=False)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    scene=Scene(a.data,a.block,a.patch,a.split_seed)
    train=Patches(scene,'train',a.noise,a.seed,a.noise_model);val=Patches(scene,'val')
    generator=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,generator=generator)
    validation=DataLoader(val,batch_size=a.batch_size)
    # CPU is the installed, verified runtime. Record device explicitly.
    early_fusion=(a.method=='early_fusion')
    if a.method=='published_reimplementation':
        model=PublishedReimplementation(scene.h.shape[2],scene.classes)
    else:
        model=Classifier(scene.h.shape[2],scene.classes,early_fusion=early_fusion)
    teacher=copy.deepcopy(model).eval()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    manifest={'stage':'development_validation_only','method_candidate':a.method,
              'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
              'protocol':scene.manifest,'train_source_sha256':digest(__file__),
              'protocol_source_sha256':digest(Path(__file__).with_name('protocol.py')),
              'torch':torch.__version__,'numpy':np.__version__,'device':'cpu',
              'training_noise_changed':int(train.corrupt.sum()),'training_samples':len(train),
              'noise_eligible_samples':int(train.eligible.sum()),
              'noise_effective_rate':float(train.corrupt.sum()/max(int(train.eligible.sum()),1)),
              'noise_flag_use':'diagnostics only; never supplied to objective',
              'test_evaluation':False,'parameters':sum(x.numel() for x in model.parameters())}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # Per-sample loss EMA is the reliability signal. It is updated only from
    # the observed (possibly corrupted) training label and never from clean
    # labels or the held-out masks.
    loss_ema=np.full(len(train), np.log(scene.classes), dtype=np.float32)
    best=-1;history=[]; start=time.time()
    for epoch in range(1,a.epochs+1):
        model.train();total=0.;qvalues=[]
        for h,d,y,i in loader:
            if a.ablation=='hsi_only': d.zero_()
            elif a.ablation=='lidar_only': h.zero_()
            predictions=model(h,d)
            q=torch.ones(len(y));target=None
            if a.method not in ('ce','early_fusion','published_reimplementation') and epoch>a.warmup:
                with torch.no_grad():
                    med=float(np.median(loss_ema))
                    q=torch.from_numpy(np.exp(-loss_ema[i.numpy()]/max(med,1e-3))).float().clamp(.05,1.)
                    if a.method in ('boundary_reliability','gce_boundary'):
                        q=q*(1-.30*boundary_proxy(h,d).cpu())
            loss=0.
            ce_values=[]
            for factor,z in zip([1.,.25,.25],predictions):
                ce=F.cross_entropy(z,y,reduction='none')
                ce_values.append(ce.detach())
                if a.method=='gce_boundary':
                    # Generalized cross entropy (q=0.7) limits the influence
                    # of high-loss corrupted labels while retaining class
                    # information from confident examples.
                    prob=z.softmax(1).gather(1,y[:,None]).squeeze(1).clamp_min(1e-6)
                    per_sample=(1-prob.pow(a.gce_q))/a.gce_q
                else:
                    per_sample=ce
                loss=loss+factor*(q.to(ce.device)*per_sample).sum()/q.sum().clamp_min(1e-6)
            with torch.no_grad():
                observed=ce_values[0].cpu().numpy()
                ids=i.numpy()
                loss_ema[ids]=.9*loss_ema[ids]+.1*observed
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            with torch.no_grad():
                for t,s in zip(teacher.parameters(),model.parameters()): t.mul_(.99).add_(s,alpha=.01)
            total+=float(loss.detach())*len(y);qvalues.extend(q.tolist())
        result=evaluate(model,validation,scene)
        row={'epoch':epoch,'training_loss':total/len(train),'reliability_mean':float(np.mean(qvalues)),
             'validation':result,'seconds':time.time()-start}
        history.append(row)
        with (a.out/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({'method':a.method,'epoch':epoch,'val_oa':result['oa'],
                          'val_aa_present':result['aa_present_classes'],'seconds':row['seconds']}),flush=True)
        if result['oa']>best:
            best=result['oa'];torch.save({'student':model.state_dict(),'teacher':teacher.state_dict(),
                                       'epoch':epoch,'validation':result},a.out/'best.pt')
    torch.save({'student':model.state_dict(),'teacher':teacher.state_dict()},a.out/'last.pt')
    (a.out/'complete.json').write_text(json.dumps({'best_validation_oa':best,'epochs':len(history),
                     'test_evaluation':False,'seconds':time.time()-start},indent=2),encoding='utf-8')


if __name__=='__main__':main()
