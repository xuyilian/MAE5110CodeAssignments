"""Minimum/maximum step trajectories with analytical stance and RK4 standing."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from validate_uniform_velocity import choose_action
from two_interval_policy import (capture_interval,SMALL_UPPER,G,L,GAMMA,
                                 ALPHA_MIN,ALPHA_MAX,transition)
from plot_analytical_capture import boundaries

OUT=Path(__file__).resolve().parents[1]/'figures'
KP,KD=240.,80.


def torque(state):
    return float(np.clip(-G*np.sin(state[0])-KP*state[0]-KD*state[1],-.1*G,.05*G))


def rhs(state):
    return np.array([state[1],G*np.sin(state[0])+torque(state)])


def rk4(state,dt):
    a=rhs(state); b=rhs(state+dt*a/2); c=rhs(state+dt*b/2); d=rhs(state+dt*c)
    return state+dt*(a+2*b+2*c+d)/6


def run(mode,dt=.0005,points=1501):
    time=[0.]; theta=[0.]; speed=[4.]; torques=[0.]; impacts=[]
    def passive(end):
        angles=np.linspace(theta[-1],end,points)
        velocities=np.sqrt(speed[-1]**2+2*G*(np.cos(angles[0])-np.cos(angles)))
        durations=np.diff(angles)*.5*(1/velocities[:-1]+1/velocities[1:])
        times=time[-1]+np.cumsum(durations)
        time.extend(times.tolist());theta.extend(angles[1:].tolist());speed.extend(velocities[1:].tolist());torques.extend([0.]*(points-1))
    for k in range(1,21):
        before_section=speed[-1]
        alpha=float(choose_action(np.array([before_section]))[0]) if mode=='minimum' else ALPHA_MIN
        passive(GAMMA+alpha)
        before=[theta[-1],speed[-1]]
        after=[GAMMA-alpha,np.cos(2*alpha)*speed[-1]]
        time.append(time[-1]);theta.append(after[0]);speed.append(after[1]);torques.append(0.)
        low,high=boundaries(np.array([after[0]]))
        captured=bool(low[0]+1e-10<after[1]<high[0]-1e-10)
        impacts.append({'step':k,'time':time[-1],'section_velocity':before_section,'alpha_rad':alpha,'pre':before,'post':after,'captured':captured})
        if captured:break
        passive(0.)
    else:raise RuntimeError('No capture')
    capture_index=len(time)-1
    capture_time=time[-1]; capture_state=[theta[-1],speed[-1]]
    state=np.array(capture_state); torques[-1]=torque(state); hold=0.; first_settled=None
    for i in range(round(12/dt)):
        state=rk4(state,dt)
        time.append(capture_time+(i+1)*dt);theta.append(state[0]);speed.append(state[1]);torques.append(torque(state))
        if np.max(np.abs(state))<1e-6:
            if hold==0:first_settled=time[-1]
            hold+=dt
            if hold>=.5:break
        else:hold=0.;first_settled=None
    else:raise RuntimeError('Standing convergence tolerance not reached')
    states=np.column_stack([theta,speed]);times=np.array(time);tau=np.array(torques)
    low,high=boundaries(states[capture_index:,0])
    controlled=states[capture_index:,1]
    assert np.all(controlled>low-1e-9) and np.all(controlled<high+1e-9)
    assert tau.min()>=-.981-1e-12 and tau.max()<=.4905+1e-12
    summary={'mode':mode,'footstrikes':len(impacts),'capture_time_s':capture_time,'capture_index':capture_index,'capture_state':capture_state,
             'settling_time_s':first_settled,'standing_duration_to_tolerance_s':first_settled-capture_time,
             'end_state':state.tolist(),'min_torque_Nm':float(tau.min()),'max_torque_Nm':float(tau.max()),'impacts':impacts}
    return times,states,tau,summary


def draw(results):
    fig,axes=plt.subplots(1,2,figsize=(14,7),layout='constrained')
    angles=np.linspace(-.43,.54,600);lo,hi=boundaries(angles)
    for ax,(time,state,tau,summary) in zip(axes,results):
        ax.fill_between(angles,lo,hi,color='#c7e5d9',label='Standing RoA')
        capture=summary['capture_time_s'];walking=np.arange(len(time))<summary['capture_index']
        # Plot continuous segments separately so resets are explicitly dashed arrows.
        start=0
        for event in summary['impacts']:
            stop=np.searchsorted(time,event['time'],side='left')
            ax.plot(state[start:stop+1,0],state[start:stop+1,1],c='#266d99',lw=2)
            ax.annotate('',event['post'],event['pre'],arrowprops={'arrowstyle':'->','color':'#bb642a','ls':'--','lw':1.8})
            ax.text(event['pre'][0]+.018,event['pre'][1],str(event['step']),fontsize=13,color='#98501e',va='center')
            start=stop+1
        ax.plot(state[~walking,0],state[~walking,1],c='#864090',lw=2.5,label='Ankle control')
        ax.scatter([0],[4],c='#1e4258',s=65,zorder=5)
        ax.scatter([0],[0],c='black',s=35,zorder=6)
        ax.scatter(*summary['capture_state'],c='#864090',s=65,zorder=6)
        ax.set(xlim=(-.44,.58),ylim=(-.5,4.5),title=f"{summary['mode'].capitalize()}: {summary['footstrikes']} footstrikes\n" + (r'$\alpha=\pi/7$ each step' if summary['mode']=='minimum' else r'$\alpha=\pi/8$ each step'),xlabel=r'$\theta$ (rad)',ylabel=r'$\dot\theta$ (rad/s)')
        ax.title.set_fontsize(21);ax.xaxis.label.set_fontsize(19);ax.yaxis.label.set_fontsize(19);ax.tick_params(labelsize=13)
        ax.grid(alpha=.15)
        ax.legend(loc='upper left',fontsize=12)
    fig.suptitle(r'Same initial state: $(\theta_0,\dot\theta_0)=(0,4)$',fontsize=23)
    fig.supxlabel('Blue: passive stance. Orange dashed arrows: numbered impacts. Purple: convergence to upright.',fontsize=14)
    fig.savefig(OUT/'final_phase_trajectories.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(14,10),sharex='col',layout='constrained')
    for col,(time,state,tau,summary) in enumerate(results):
        for row,values in enumerate([state[:,0],state[:,1],tau]):
            ax=axes[row,col];ax.plot(time,values,lw=2,c=['#266d99','#864090','#ba6626'][row])
            ax.axvline(summary['capture_time_s'],c='#555555',ls='--',lw=1.5)
            ax.axhline(0,c='#999999',lw=.7);ax.grid(alpha=.15);ax.tick_params(labelsize=13)
            ax.set_ylabel([r'$\theta$ (rad)',r'$\dot\theta$ (rad/s)',r'$\tau$ (N m)'][row],fontsize=18)
        axes[0,col].set_title(f"{summary['mode'].capitalize()}: {summary['footstrikes']} footstrikes",fontsize=22)
        axes[2,col].set_xlabel('Time (s)',fontsize=18)
    fig.suptitle('Walking followed by bounded ankle control',fontsize=24)
    fig.supxlabel('Dashed vertical lines: RoA entry and activation of the standing controller.',fontsize=14)
    fig.savefig(OUT/'final_time_trajectories.png',dpi=180);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results=[run(mode) for mode in ['minimum','maximum']]
    summaries=[]
    for result in results:
        t,x,u,s=result
        rt,rx,ru,rs=run(s['mode'],dt=.00025,points=3001)
        s['refinement']={'capture_time_difference_s':abs(rs['capture_time_s']-s['capture_time_s']),
                         'settling_time_difference_s':abs(rs['settling_time_s']-s['settling_time_s']),
                         'capture_state_difference':float(np.max(np.abs(np.array(rs['capture_state'])-s['capture_state'])))}
        summaries.append(s)
        np.savez(OUT/f"{s['mode']}_trajectory.npz",time=t,state=x,torque=u)
    # Universal upper envelope: alpha_min maximizes every nonterminal return.
    q=G/L
    # derivative of return radicand < 0 for every possible nonterminal state w>=SMALL_UPPER
    bound=-4*np.cos(2*ALPHA_MAX)*np.sin(2*ALPHA_MIN)*(SMALL_UPPER**2+2*q*(1-np.cos(GAMMA+ALPHA_MIN)))+2*q*np.cos(2*ALPHA_MIN)**2*np.sin(GAMMA+ALPHA_MAX)
    assert bound<0
    envelope=[4.]
    for _ in range(4):
        _,valid,next_w,_=transition(np.array([envelope[-1]]),ALPHA_MIN)
        assert valid[0]
        envelope.append(float(next_w[0]))
    assert envelope[-1]<SMALL_UPPER
    data={'initial_state':[0.,4.],'results':summaries,'maximum_proof':{'nonterminal_min_velocity':SMALL_UPPER,'upper_envelope':envelope,'radicand_derivative_upper_bound':float(bound),'maximum_footstrikes':5}}
    (OUT/'final_trajectories_summary.json').write_text(json.dumps(data,indent=2)+'\n')
    draw(results)
    print(json.dumps(data,indent=2))


if __name__=='__main__':main()
