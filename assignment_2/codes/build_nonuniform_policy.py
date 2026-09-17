"""Construct interval policy from analytical capture and return-map boundaries."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from build_lookup_policy import LIMIT, CAPTURE, ALPHA_MIN, ALPHA_MAX, G, L, GAMMA
from plot_analytical_capture import boundaries

OUT=Path(__file__).resolve().parents[1]/'figures'


def main():
    def capture_interval(a):
        low,high=boundaries(np.array([GAMMA-a]))
        c=np.cos(2*a); d=2*G/L*(1-np.cos(GAMMA+a))
        return float(np.sqrt(max(0,(low[0]/c)**2-d))),float(np.sqrt((high[0]/c)**2-d))
    _,u=capture_interval(ALPHA_MIN)
    low,b1=capture_interval(ALPHA_MAX)
    switch=(low+u)/2
    aa=np.cos(2*ALPHA_MAX)**2
    bb=2*G/L*(aa*(1-np.cos(GAMMA+ALPHA_MAX))-(1-np.cos(GAMMA-ALPHA_MAX)))
    b2=float(np.sqrt((b1*b1-bb)/aa))
    b3=float(np.sqrt((b2*b2-bb)/aa))
    edges=[CAPTURE,b1,b2]
    # Analytical inequalities certify one-step overlap and the next two preimages.
    assert CAPTURE<low<switch<u<b1<b2<LIMIT<b3
    # For w >= u, the return radicand decreases with alpha over the action range.
    # Bound its derivative from above, dropping an additional negative term.
    derivative_upper=(-4*np.cos(2*ALPHA_MAX)*np.sin(2*ALPHA_MIN)
        *(u*u+2*G/L*(1-np.cos(GAMMA+ALPHA_MIN)))
        +2*G/L*np.cos(2*ALPHA_MIN)**2*np.sin(GAMMA+ALPHA_MAX))
    assert derivative_upper<0
    # Positive lower bound for d(U(alpha)^2)/dalpha; omit another positive term.
    capture_derivative_lower=2*G/L*((np.sin(ALPHA_MIN-GAMMA)+.1)
        /np.cos(2*ALPHA_MIN)**2-np.sin(GAMMA+ALPHA_MAX))
    assert capture_derivative_lower>0
    rows=[(0,CAPTURE,0),(CAPTURE,b1,1),(b1,b2,2),(b2,LIMIT,3)]
    data={'method':'analytical nonuniform intervals','capture_threshold':CAPTURE,
          'small_angle_upper':u,'large_angle_lower':low,'switch':switch,
          'step_boundaries':[CAPTURE,b1,b2],'next_boundary':b3,
          'return_derivative_upper_bound':derivative_upper,
          'capture_derivative_lower_bound':capture_derivative_lower,
          'rows':[{'left':x,'right':y,'steps':n} for x,y,n in rows]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'nonuniform_policy.json').write_text(json.dumps(data,indent=2)+'\n')
    np.savetxt(OUT/'nonuniform_policy.csv',np.array(rows),delimiter=',',header='omega_left,omega_right,footstrikes',comments='')
    fig,axs=plt.subplots(2,1,figsize=(12,8),sharex=True,layout='constrained')
    colors=['#c9e7df','#e9bd89','#bdb0dc','#dfb9ce']
    for i,((x,y,n),color) in enumerate(zip(rows,colors),1):
        for ax in axs: ax.axvspan(x,y,color=color,alpha=.6)
        axs[1].plot([x,y],[n]*2,c='#653c84',lw=3)
        axs[0].text((x+y)/2,24.1,f'Cell {i}',ha='center',fontsize=14)
    axs[0].plot([CAPTURE,switch,switch,LIMIT],np.degrees([ALPHA_MIN,ALPHA_MIN,ALPHA_MAX,ALPHA_MAX]),c='#234a66',lw=3)
    for ax in axs:
        for x in edges: ax.axvline(x,color='#777777',ls=':',lw=1.3)
        ax.tick_params(labelsize=14)
        ax.set_xlim(0,LIMIT)
    axs[0].set_ylim(22,26.3)
    axs[0].set_ylabel(r'Action $\alpha$ (degrees)',fontsize=19)
    axs[0].set_title('Nonuniform grid: four constant-step-count intervals',fontsize=21,pad=15)
    axs[0].text(CAPTURE/2,25.5,'Stand',ha='center',fontsize=13)
    axs[1].set_ylabel('Footstrikes to RoA',fontsize=19)
    axs[1].set_yticks([0,1,2,3])
    axs[1].set_ylim(-.25,3.35)
    axs[1].set_xlabel(r'Section velocity $\omega_k$ (rad/s)',fontsize=19)
    fig.supxlabel('Cell edges: '+', '.join(f'{x:.5f}' for x in [0,*edges,LIMIT])+' rad/s\n'
                  'Use interval membership, not nearest-node lookup. Re-evaluate after each step.',fontsize=13)
    fig.savefig(OUT/'nonuniform_policy.png',dpi=180)
    plt.close(fig)
    print(json.dumps(data,indent=2))


if __name__=='__main__': main()
