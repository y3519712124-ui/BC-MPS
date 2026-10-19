"""Independent experiments for adaptive cross-modal reliability gating.

The gate combines loss-history risk, HSI/LiDAR auxiliary-head disagreement,
and input complexity.  Complexity is an interaction term and cannot by itself
down-weight a sample.  This file deliberately writes to a new result group.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse, csv, gc, hashlib, json, random, time
import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import rankdata
from protocol import Scene, Patches, digest
from train import Classifier, boundary_proxy

ROOT=Path(__file__).resolve().parents[1]
GROUP='adaptive_reliability_v4'

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)

def cache(scene, subset, device, batch=256):
    ds=Patches(scene,subset); coords=ds.coords; n=len(coords); s=scene.patch; r=s//2
    h=torch.empty((n,scene.h.shape[2],s,s),device=device); d=torch.empty((n,1,s,s),device=device)
    for st in range(0,n,batch):
        c=coords[st:st+batch]; yy=c[:,0,None,None]+np.arange(-r,r+1)[None,:,None]
        xx=c[:,1,None,None]+np.arange(-r,r+1)[None,None,:]
        h[st:st+len(c)].copy_(torch.from_numpy(scene.h[yy,xx].transpose(0,3,1,2).copy()))
        d[st:st+len(c)].copy_(torch.from_numpy(scene.d[yy,xx][:,None].copy()))
    y=torch.from_numpy(ds.clean).long().to(device); b=torch.empty(n,device=device)
    with torch.no_grad():
        for st in range(0,n,batch): b[st:st+batch]=boundary_proxy(h[st:st+batch],d[st:st+batch])
    return {'h':h,'d':d,'y':y,'coords':coords,'b':b,'n':n}

def metrics(p,y,k):
    p=np.asarray(p,float); y=np.asarray(y); pred=p.argmax(1); cm=np.bincount(y*k+pred,minlength=k*k).reshape(k,k)
    cnt=cm.sum(1); present=cnt>0; rec=np.divide(cm.diagonal(),cnt,out=np.zeros(k),where=present); n=len(y)
    oa=float((pred==y).mean()); chance=float((cnt*cm.sum(0)).sum()/n**2); conf=p.max(1); ok=pred==y; ece=0.
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        take=(conf>lo)&(conf<=hi)
        if take.any(): ece+=take.mean()*abs(ok[take].mean()-conf[take].mean())
    oh=np.eye(k)[y]
    return {'oa':oa*100,'aa':float(rec[present].mean()*100),'kappa':100*(oa-chance)/(1-chance) if chance<1 else 0.,
            'ece':ece*100,'nll':float(-np.log(p[np.arange(n),y].clip(1e-7)).mean()),'brier':float(np.square(p-oh).sum(1).mean()),
            'pixels':n,'present_classes':int(present.sum()),'per_class_recall':[float(v*100) if present[i] else None for i,v in enumerate(rec)],'confusion':cm.tolist()}

@torch.no_grad()
def predict(model,bundle,batch):
    model.eval(); out=[]
    for st in range(0,bundle['n'],batch): out.append(model(bundle['h'][st:st+batch],bundle['d'][st:st+batch])[0].softmax(1).cpu().numpy())
    return np.concatenate(out)

def gate_weights(loss_ema, aux_h, aux_d, y, complexity, variant):
    """Return bounded weights and diagnostics; all inputs are detached."""
    med=torch.quantile(loss_ema,.5).clamp_min(.001)
    r=(loss_ema/med).clamp(0,6.)/6.                 # persistent loss risk
    ph=aux_h.softmax(1).gather(1,y[:,None]).squeeze(1)
    pd=aux_d.softmax(1).gather(1,y[:,None]).squeeze(1)
    u=(ph-pd).abs().clamp(0,1.)                    # modality conflict
    q=complexity.clamp(0,1.)                       # input complexity
    if variant=='no_consistency': u.zero_()
    if variant=='no_complexity': q.zero_()
    # Complexity has no independent coefficient: it only strengthens joint risk.
    risk=(0.50*r + 0.35*u + 0.15*(r*u*q)).clamp(0,1.)
    w=(1.0-0.55*torch.sigmoid(6*(risk-.45))).clamp(.30,1.)
    return w, {'risk_mean':float(risk.mean()),'weight_mean':float(w.mean()),'complexity_mean':float(q.mean()),'disagreement_mean':float(u.mean())}

def noise_targets(scene,train,seed,noise):
    clean=train['y'].cpu().numpy(); rng=np.random.default_rng(seed); changed=rng.random(len(clean))<noise
    offsets=rng.integers(1,scene.classes,len(clean)); observed=clean.copy(); observed[changed]=(clean[changed]+offsets[changed])%scene.classes
    return torch.from_numpy(observed).long().to(train['y'].device),changed

def run(scene,train,val,spec,device,max_epochs,patience,batch):
    ds,split,method,seed,noise,variant=[spec[x] for x in ('dataset','split_seed','method','seed','noise','variant')]
    name=f'{ds}_split{split}_{method}_s{seed}_n{noise:.2f}_{variant}'; out=ROOT/'runs'/GROUP/name
    if (out/'test_metrics.json').exists(): return json.loads((out/'test_metrics.json').read_text())
    if out.exists(): raise RuntimeError(f'incomplete run preserved: {out}')
    out.mkdir(parents=True); seed_everything(seed); model=Classifier(scene.h.shape[2],scene.classes).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001); observed,changed=noise_targets(scene,train,seed,noise)
    loss_ema=torch.full((train['n'],),float(np.log(scene.classes)),device=device); best=-1.;best_epoch=0;hist=[]; started=time.time()
    manifest={'spec':spec,'protocol':scene.manifest,'method_definition':'adaptive gate: persistent loss, auxiliary disagreement, gradient interaction','source_sha256':digest(__file__),'model_source_sha256':digest(ROOT/'validated_protocol/train.py'),'test_used_for_selection':False,'training_noise_count':int(changed.sum())}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for epoch in range(1,max_epochs+1):
        model.train(); total=0.; diagnostics=[]; order=torch.randperm(train['n'],device=device)
        for st in range(0,len(order),batch):
            ids=order[st:st+batch]; h=train['h'][ids]; d=train['d'][ids]; y=observed[ids]; pred=model(h,d)
            with torch.no_grad():
                if epoch>5: w,diag=gate_weights(loss_ema[ids], pred[1].detach(),pred[2].detach(),y,train['b'][ids],variant)
                else: w=torch.ones(len(ids),device=device); diag={'risk_mean':0.,'weight_mean':1.,'complexity_mean':float(train['b'][ids].mean()),'disagreement_mean':0.}
            each=[]
            for alpha,logits in zip((1.,.25,.25),pred):
                p=logits.softmax(1).gather(1,y[:,None]).squeeze(1).clamp_min(1e-6); each.append(alpha*((1-p.pow(.7))/.7))
            loss=sum((w*x).sum()/w.sum().clamp_min(1e-6) for x in each)
            if spec['method']=='cmct':
                # Confidence-gated cross-modal co-teaching.  The more confident
                # auxiliary branch teaches the fused head; no label is used to
                # choose the teacher and the target is detached.
                pf, ph, pd = [z.softmax(1) for z in pred]
                ch, cd = ph.max(1).values, pd.max(1).values
                teacher=torch.where((ch>=cd)[:,None], ph.detach(), pd.detach())
                confidence=torch.maximum(ch,cd).detach()
                agreement=(1-(ph-pd).abs().mean(1)).detach().clamp(0,1)
                # Apply distillation only to high-agreement patches; preserve
                # complementary evidence when the modalities disagree.
                reliability=(w*confidence*agreement.pow(4)).detach()
                consistency=F.kl_div(pf.clamp_min(1e-6).log(),teacher,reduction='none').sum(1)
                loss=loss+0.05*(reliability*consistency).sum()/reliability.sum().clamp_min(1e-6)
            fused=F.cross_entropy(pred[0],y,reduction='none')
            with torch.no_grad(): loss_ema[ids]=.9*loss_ema[ids]+.1*fused.detach()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
            total+=float(loss.detach())*len(ids); diagnostics.append(diag)
        v=metrics(predict(model,val,batch),val['y'].cpu().numpy(),scene.classes); row={'epoch':epoch,'validation':v,'training_loss':total/train['n'],'gate':{k:float(np.mean([d[k] for d in diagnostics])) for k in diagnostics[0]},'seconds':time.time()-started}; hist.append(row)
        with (out/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if v['oa']>best: best=v['oa'];best_epoch=epoch; torch.save({'student':model.state_dict(),'epoch':epoch,'validation':v},out/'best.pt'); np.savez_compressed(out/'best_reliability.npz',loss_ema=loss_ema.cpu().numpy(),corrupt=changed)
        if epoch>=30 and epoch-best_epoch>=patience: break
    ck=torch.load(out/'best.pt',map_location=device,weights_only=False); model.load_state_dict(ck['student']); test=cache(scene,'test',torch.device('cpu')); p=predict(model,{'h':test['h'].to(device),'d':test['d'].to(device),'n':test['n']},batch); result=metrics(p,test['y'].numpy(),scene.classes)
    e=np.load(out/'best_reliability.npz')['loss_ema']; w=np.exp(-e/max(np.median(e),.001)).clip(.05,1.); auc=None
    if changed.any() and (~changed).any():
        ranks=rankdata(-w); n1=int(changed.sum()); n0=len(changed)-n1; auc=float((ranks[changed].sum()-n1*(n1+1)/2)/(n1*n0))
    comp={'epochs':len(hist),'best_epoch':best_epoch,'training_seconds':time.time()-started}; diag={'reliability_noise_auc':auc,'weight_mean':float(w.mean())}
    rec={'spec':spec,'manifest':manifest,'completion':comp,'test':result,'diagnostics':diag,'run_dir':str(out)}; (out/'test_metrics.json').write_text(json.dumps(rec,indent=2),encoding='utf-8'); return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--split-seed',type=int,default=20261031); ap.add_argument('--method',choices=['adaptive','cmct'],default='cmct'); ap.add_argument('--variant',choices=['full','no_consistency','no_complexity'],default='full'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--epochs',type=int,default=35); ap.add_argument('--device',choices=['cuda','cpu'],default='cuda'); ap.add_argument('--batch',type=int,default=256); ap.add_argument('--noise',type=float,default=.4)
    a=ap.parse_args(); device=torch.device(a.device); scene=Scene(a.data,64,11,a.split_seed); train=cache(scene,'train',device); val=cache(scene,'val',device); spec={'dataset':'muufl','split_seed':a.split_seed,'method':a.method,'seed':a.seed,'noise':a.noise,'variant':a.variant}; rec=run(scene,train,val,spec,device,a.epochs,20,a.batch); print(json.dumps({'complete':rec['run_dir'],'test_oa':rec['test']['oa'],'test_aa':rec['test']['aa']}))

if __name__=='__main__': main()
