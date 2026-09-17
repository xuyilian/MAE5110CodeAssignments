"""Compare uniform lookup grids on a fixed exact-map test set."""
import json
from pathlib import Path
import numpy as np
from build_lookup_policy import build, polish, rollout, LIMIT, CAPTURE, transition, ALPHA_MAX, G, L, GAMMA
from plot_analytical_capture import boundaries

OUT=Path(__file__).resolve().parents[1]/'figures'

def tests():
    a=ALPHA_MAX
    upper=float(boundaries(np.array([GAMMA-a]))[1][0])
    threshold1=np.sqrt((upper-1e-10)**2/np.cos(2*a)**2-2*G/L*(1-np.cos(GAMMA+a)))
    A=np.cos(2*a)**2
    B=A*2*G/L*(1-np.cos(GAMMA+a))-2*G/L*(1-np.cos(GAMMA-a))
    threshold2=np.sqrt((threshold1**2-B)/A)
    # Fixed probes on both sides of analytically located transition candidates.
    offsets=np.array([-1e-3,-1e-4,-1e-5,-1e-6,1e-6,1e-5,1e-4,1e-3])
    points=np.unique(np.r_[np.linspace(0,LIMIT,10001),*(t+offsets for t in [CAPTURE,threshold1,threshold2])])
    return points,[CAPTURE,threshold1,threshold2]

def evaluate(nw,na,points):
    w,a,v,p,c,it=build(nw,na)
    del c
    v,p,c=polish(w,a,p)
    result=rollout(points,w,p,max_steps=20)
    return result,w,a,v,p,c

if __name__=='__main__':
    points,thresholds=tests()
    print('tests',len(points),'thresholds',thresholds,flush=True)
    previous=None
    for nw,na in [(2401,401),(4801,401),(4801,801),(9601,801)]:
        result,w,a,v,p,c=evaluate(nw,na,points)
        print(nw,na,'fail',int(np.sum(result<0)),'diff previous',None if previous is None else int(np.sum(result!=previous)),flush=True)
        np.savez(OUT/f'reference_{nw}_{na}.npz',result=result,omega=w,alpha=a,value=v,policy=p)
        previous=result
        del c

    rows=[]
    best=[]
    reference=previous
    for cells in range(4,401):
        for nw in range(2,cells//2+1):
            if cells%nw: continue
            na=cells//nw
            result,*unused=evaluate(nw,na,points)
            row={'nw':nw,'na':na,'cells':cells,'failures':int(np.sum(result<0)),
                 'disagreements':int(np.sum((result>=0)&(result!=reference)))}
            rows.append(row)
            if row['failures']==0 and row['disagreements']==0: best.append(row)
        if best: break
    (OUT/'grid_search_results.json').write_text(json.dumps(
        {'test_count':len(points),'thresholds':thresholds,'reference_failures':int(np.sum(reference<0)),
         'results':rows,'best':best},indent=2)+'\n')
    print('Smallest passing tables:',best,flush=True)
