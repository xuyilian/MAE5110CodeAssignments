"""Plot passive return curves, stopping outcomes, and the saved lookup policy."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from build_lookup_policy import transition, CAPTURE, LIMIT, ALPHA_MIN, ALPHA_MAX

OUT=Path(__file__).resolve().parents[1]/'figures'


def main():
    table=np.load(OUT/'lookup_policy.npz')
    w=table['omega']
    policy=table['policy']
    angles=[ALPHA_MIN,(ALPHA_MIN+ALPHA_MAX)/2,ALPHA_MAX]
    colors=['#c26a20','#7046a0','#25816f']
    fig,(ax,events)=plt.subplots(2,1,figsize=(12,10),sharex=True,
                               gridspec_kw={'height_ratios':[3,1.3]},layout='constrained')
    event_rows=[]
    for alpha,color in zip(angles,colors):
        capture,valid,nxt,_=transition(w,alpha)
        active=w>=CAPTURE
        curve=np.where(active & valid,nxt,np.nan)
        label=rf'$\alpha={np.degrees(alpha):.2f}^\circ$'
        ax.plot(w,curve,color=color,lw=2.7,label=label)
        event_rows.append((capture & active,(~capture)&(~valid)&active,color,label))
    capture,valid,nxt,_=transition(w,policy)
    ax.plot(w,np.where(valid,nxt,np.nan),color='#182e43',lw=2.2,ls='--',label='Selected lookup policy')
    event_rows.append((capture & (w>=CAPTURE),np.isfinite(policy)&(~capture)&(~valid),'#182e43','Lookup policy'))
    ax.plot([0,LIMIT],[0,LIMIT],color='#999999',ls=':',lw=2,label=r'$\omega_{k+1}=\omega_k$')
    ax.axvspan(0,CAPTURE,color='#d7ede7',label='Already in standing RoA')
    ax.set_ylim(0,LIMIT)
    ax.set_xlim(0,LIMIT)
    ax.set_ylabel(r'Next velocity $\omega_{k+1}$ (rad/s)',fontsize=20)
    ax.set_title('Poincaré return map and stopping outcomes',fontsize=25,pad=16)
    ax.legend(loc='upper left',fontsize=13,framealpha=.95)
    ax.text(.97,.06,'Below the diagonal: velocity decreases',transform=ax.transAxes,
            ha='right',fontsize=14,bbox={'facecolor':'white','edgecolor':'none'})
    for row,(captured,failed,color,label) in enumerate(event_rows):
        events.plot(w,np.where(captured,row,np.nan),color=color,lw=8,solid_capstyle='butt')
        events.plot(w,np.where(failed,row,np.nan),color='#a5a5a5',lw=8,solid_capstyle='butt')
    events.axvspan(0,CAPTURE,color='#d7ede7')
    events.set_yticks(range(len(event_rows)),[r[3] for r in event_rows])
    events.set_ylim(3.6,-.6)
    events.set_title('Terminal outcomes before the next section crossing',fontsize=19,pad=12)
    events.set_xlabel(r'Current section velocity $\omega_k$ (rad/s), at $\theta=0$',fontsize=20)
    for axis in [ax,events]:
        axis.tick_params(labelsize=14)
        axis.grid(alpha=.18)
    fig.supxlabel('Lower panel: colored = RoA entry after one impact; gray = no forward return or capture.\n'
                  'Blank = a return is shown above. Pale green = zero-step capture. No fictitious return is assigned to capture.',fontsize=13)
    fig.savefig(OUT/'return_map.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
