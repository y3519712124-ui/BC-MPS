"""Spatially blocked conformal prediction sets for HSI--LiDAR outputs.

Calibration uses validation tiles only. Test labels are read only after all
thresholds are fixed, and are used for coverage/risk reporting.
"""
from pathlib import Path
import argparse, json, math
import numpy as np
import torch
from protocol import Scene
from train import Classifier
from adaptive_reliability_experiments import cache

def outputs(model,bundle,device,batch=256):
    out=[]
    with torch.no_grad():
        for st in range(0,bundle['n'],batch):
            z=model(bundle['h'][st:st+batch].to(device),bundle['d'][st:st+batch].to(device))
            out.append(np.stack([x.softmax(1).cpu().numpy() for x in z]))
    return np.concatenate(out,axis=1)

def adaptive_probs(o,tau=.12):
    pf,ph,pd=o; ch=ph.max(1); cd=pd.max(1)
    a=1/(1+np.exp(-(ch-cd)/tau))
    return .75*pf+.25*(a[:,None]*ph+(1-a[:,None])*pd)

def quantile_higher(x,level):
    if len(x)==0: raise ValueError('empty calibration scores')
    # Empirical capped upper order statistic with the (n+1) correction.
    # An out-of-range rank uses the largest observed score; standard
    # split conformal would use infinity. This cap alone gives no guarantee.
    rank=min(len(x)-1,max(0,int(math.ceil((len(x)+1)*level))-1))
    return float(np.sort(np.asarray(x))[rank])

def sets_and_metrics(p,y,q):
    take=p>=1-q; true=take[np.arange(len(y)),y]
    size=take.sum(1); singleton=(size==1)&true
    pred=p.argmax(1)
    return {'coverage':float(true.mean()),'mean_set_size':float(size.mean()),'singleton_accuracy':float(singleton.sum()/max((size==1).sum(),1)),
            'singleton_fraction':float((size==1).mean()),'empty_fraction':float((size==0).mean()),'oa_argmax':float((pred==y).mean()),'q':float(q)}

def block_summary(p,y,tile,q_by_tile):
    vals=[]
    for t in sorted(np.unique(tile)):
        ix=tile==t
        if ix.any(): vals.append(sets_and_metrics(p[ix],y[ix],q_by_tile[t])['coverage'])
    return {'block_coverage_mean':float(np.mean(vals)),'block_coverage_min':float(np.min(vals)),'block_coverage_sd':float(np.std(vals)),'blocks':len(vals)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--split-seed',type=int,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--alpha',type=float,default=.10); ap.add_argument('--device',default='cuda'); a=ap.parse_args()
    device=torch.device(a.device); scene=Scene(a.data,64,11,a.split_seed); val=cache(scene,'val',device); test=cache(scene,'test',device)
    model=Classifier(scene.h.shape[2],scene.classes).to(device); ck=torch.load(a.checkpoint,map_location=device,weights_only=False); model.load_state_dict(ck['student']); model.eval()
    ov=outputs(model,val,device); ot=outputs(model,test,device); yv=val['y'].cpu().numpy(); yt=test['y'].cpu().numpy(); tv=scene.tile[tuple(val['coords'].T)]; tt=scene.tile[tuple(test['coords'].T)]
    predictors={'fused':(ov[0],ot[0]),'hsi':(ov[1],ot[1]),'lidar':(ov[2],ot[2]),'adaptive':(adaptive_probs(ov),adaptive_probs(ot))}
    result={'split_seed':a.split_seed,'checkpoint':str(a.checkpoint),'alpha':a.alpha,'calibration_pixels':len(yv),'calibration_blocks':len(np.unique(tv)),
      'method_definition':'validation-tile capped score quantiles; corrected capped upper rank min(m, ceil((m+1)*0.75)) across tile thresholds; maximum when m=3'}
    result['predictors']={}
    for name,(pv,pt) in predictors.items():
        scores=1-pv[np.arange(len(yv)),yv]; level=1-a.alpha; q_global=quantile_higher(scores,level)
        block_q={int(t):quantile_higher(scores[tv==t],level) for t in np.unique(tv)}; q_block_q75=quantile_higher(list(block_q.values()),.75)
        g=sets_and_metrics(pt,yt,q_global); g.update(block_summary(pt,yt,tt,{int(t):q_global for t in np.unique(tt)}))
        b=sets_and_metrics(pt,yt,q_block_q75); b.update(block_summary(pt,yt,tt,{int(t):q_block_q75 for t in np.unique(tt)}))
        result['predictors'][name]={'global_conformal':g,'blocked_conformal':b,'thresholds':{'global':q_global,'block_q75':q_block_q75}}
    # Emit one compact JSON object per line so the recorded file is valid
    # JSONL and can be consumed by standard line-oriented tooling.
    print(json.dumps(result, separators=(',', ':')))
if __name__=='__main__': main()
