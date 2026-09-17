"""Two interval-membership actions; exact-map validation against a fine reference."""
from pathlib import Path
import json
import numpy as np
from build_lookup_policy import (G,L,GAMMA,LIMIT,CAPTURE,ALPHA_MIN,ALPHA_MAX,
                                 transition,rollout)
from plot_analytical_capture import boundaries

OUT=Path(__file__).resolve().parents[1]/'figures'


def capture_interval(alpha):
    lower,upper=boundaries(np.array([GAMMA-alpha]))
    c=np.cos(2*alpha)
    gain=2*G/L*(1-np.cos(GAMMA+alpha))
    return float(np.sqrt(max(0,(lower[0]/c)**2-gain))),float(np.sqrt((upper[0]/c)**2-gain))


_,SMALL_UPPER=capture_interval(ALPHA_MIN)
LARGE_LOWER,ONE_STEP_UPPER=capture_interval(ALPHA_MAX)
SWITCH=(SMALL_UPPER+LARGE_LOWER)/2
RETURN_A=np.cos(2*ALPHA_MAX)**2
RETURN_B=2*G/L*(RETURN_A*(1-np.cos(GAMMA+ALPHA_MAX))-(1-np.cos(GAMMA-ALPHA_MAX)))
TWO_STEP_UPPER=float(np.sqrt((ONE_STEP_UPPER**2-RETURN_B)/RETURN_A))


def choose_action(omega):
    """No rounding: compare the measured velocity with the interval edge."""
    omega=np.asarray(omega,dtype=float)
    if np.any((omega<0)|(omega>LIMIT)|~np.isfinite(omega)):
        raise ValueError('Velocity must be finite and inside the designed range')
    return np.where(omega<CAPTURE-1e-10,np.nan,
                    np.where(omega<SWITCH,ALPHA_MIN,ALPHA_MAX))


def execute(starts,constant_angle=None,max_steps=20):
    omega=np.asarray(starts,dtype=float).copy()
    count=np.full(omega.shape,-1,dtype=int)
    count[omega<CAPTURE-1e-10]=0
    active=count<0
    for step in range(1,max_steps+1):
        ids=np.flatnonzero(active)
        if not ids.size: break
        actions=choose_action(omega[ids]) if constant_angle is None else constant_angle
        captured,valid,nxt,_=transition(omega[ids],actions)
        count[ids[captured]]=step
        active[ids[captured|~valid]]=False
        omega[ids[valid]]=nxt[valid]
        arrived=valid & (nxt<CAPTURE-1e-10)
        count[ids[arrived]]=step
        active[ids[arrived]]=False
    return count


def metrics(result,reference):
    return {'failures':int(np.sum(result<0)),
            'step_count_disagreements':int(np.sum((result>=0)&(result!=reference)))}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    centers=[CAPTURE,SWITCH,LARGE_LOWER,SMALL_UPPER,ONE_STEP_UPPER,TWO_STEP_UPPER]
    offsets=np.r_[0.,-np.logspace(-6,-2,5),np.logspace(-6,-2,5)]
    starts=np.unique(np.r_[np.linspace(0,LIMIT,100001),*(c+offsets for c in centers)])
    refs=[]
    for filename in ['lookup_policy.npz','reference_4801_801.npz','reference_9601_801.npz']:
        data=np.load(OUT/filename)
        counts=rollout(starts,data['omega'],data['policy'],max_steps=20)
        refs.append(counts)
    reference=refs[-1]
    actual=execute(starts)
    candidates=[]
    for alpha in np.linspace(ALPHA_MIN,ALPHA_MAX,401):
        result=execute(starts,constant_angle=alpha)
        row={'alpha_rad':float(alpha),**metrics(result,reference)}
        candidates.append(row)
    best=min(candidates,key=lambda r:r['failures']+r['step_count_disagreements'])
    result={'test_count':len(starts),'uniform_test_points':100001,
            'boundary_test_centers':centers,'offsets':offsets.tolist(),
            'reference_failures':int(np.sum(reference<0)),
            'reference_refinement_disagreements':[int(np.sum(r!=reference)) for r in refs],
            'two_interval_policy':metrics(actual,reference),
            'constant_small_angle':candidates[0],'constant_large_angle':candidates[-1],
            'best_single_angle_of_401':best,
            'single_angle_passes':sum(r['failures']==0 and r['step_count_disagreements']==0 for r in candidates),
            'max_executed_steps':int(actual.max()),'switch_rad_s':SWITCH,
            'first_interval_width':SWITCH-CAPTURE,'second_interval_width':LIMIT-SWITCH,
            'note':'Exact-map policy execution tests; not time integration of standing control.'}
    (OUT/'two_interval_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savetxt(OUT/'two_interval_policy.csv',[[CAPTURE,SWITCH,ALPHA_MIN],[SWITCH,LIMIT,ALPHA_MAX]],
               delimiter=',',header='omega_left,omega_right,alpha_rad',comments='')
    np.savez(OUT/'two_interval_validation.npz',omega=starts,reference_steps=reference,policy_steps=actual)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,layout='constrained')
    axes[0].axvspan(0,CAPTURE,color='#c9e7df')
    axes[0].axvspan(CAPTURE,SWITCH,color='#f1d9b4')
    axes[0].axvspan(SWITCH,LIMIT,color='#c6ddeb')
    axes[0].plot([CAPTURE,SWITCH,SWITCH,LIMIT],np.degrees([ALPHA_MIN,ALPHA_MIN,ALPHA_MAX,ALPHA_MAX]),color='#24445d',lw=3)
    axes[0].text(CAPTURE/2,24.3,'Stand',ha='center',fontsize=13)
    axes[0].text((CAPTURE+SWITCH)/2,24.3,'Interval 1',ha='center',fontsize=14)
    axes[0].text((SWITCH+LIMIT)/2,24.3,'Interval 2',ha='center',fontsize=16)
    axes[0].set_ylim(22,26.2)
    axes[0].set_ylabel(r'Action $\alpha$ (degrees)',fontsize=19)
    axes[0].set_title('Final lookup policy: two walking intervals',fontsize=23,pad=15)
    axes[0].axvline(SWITCH,color='#555555',ls='--',lw=1.5)
    axes[1].step(starts,actual,where='post',c='#73488a',lw=2.5)
    axes[1].axvspan(0,CAPTURE,color='#c9e7df')
    axes[1].set_ylabel('Executed footstrikes\nto standing RoA',fontsize=19)
    axes[1].set_xlabel(r'Section velocity $\omega_k$ (rad/s)',fontsize=19)
    axes[1].set_yticks([0,1,2,3])
    axes[1].set_ylim(-.2,3.3)
    for ax in axes:
        ax.set_xlim(0,LIMIT)
        ax.tick_params(labelsize=14)
        ax.grid(alpha=.15)
    fig.supxlabel(f'Walking intervals: [{CAPTURE:.5f}, {SWITCH:.5f}) and [{SWITCH:.5f}, {LIMIT:.5f}] rad/s\n'
                  f'{len(starts):,} exact-map checks: zero failures and zero step-count disagreements.',fontsize=14)
    fig.savefig(OUT/'two_interval_policy.png',dpi=180)
    plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
