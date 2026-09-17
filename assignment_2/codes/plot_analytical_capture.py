"""Plot analytical capture bounds; no trajectory simulations are used."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'figures'
G, L, A, B = 9.81, 1.0, .1, .05
GAMMA, ALPHA_MIN, ALPHA_MAX = .06, np.pi/8, np.pi/7
THETA_MIN, THETA_MAX = GAMMA-ALPHA_MIN, GAMMA+ALPHA_MAX
left, right = -np.arcsin(B), np.arcsin(A)


def boundaries(theta):
    upper_energy = 2*G/L*(np.cos(right)-np.cos(theta)+A*(right-theta))
    lower_energy = 2*G/L*(np.cos(left)-np.cos(theta)+B*(theta-left))
    upper = np.where(theta <= right, 1., -1.)*np.sqrt(np.maximum(upper_energy, 0))
    lower = np.where(theta >= left, -1., 1.)*np.sqrt(np.maximum(lower_energy, 0))
    return lower, upper


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(THETA_MIN, THETA_MAX, 2001)
    lower, upper = boundaries(theta)
    low0, high0 = boundaries(np.array([0.]))
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(left=.075, right=.98, bottom=.29, top=.84, wspace=.27)
    for ax in axes:
        ax.fill_between(theta, lower, upper, color='#acdacf', alpha=.85,
                        label='Capture region (interior)')
        ax.plot(theta, upper, c='#b35b16', lw=3.5, label=r'Upper boundary: $u=-0.1$')
        ax.plot(theta, lower, c='#7049a1', lw=3.5, label=r'Lower boundary: $u=+0.05$')
        ax.axvline(left, c='#777777', ls=':', lw=1.8)
        ax.axvline(right, c='#777777', ls=':', lw=1.8)
        ax.axhline(0, c='#aaaaaa', lw=.8)
        ax.axvline(0, c='#aaaaaa', lw=.8)
        ax.scatter([left,right],[0,0],facecolors='white',edgecolors='#444444',s=95,zorder=6)
        ax.scatter([0],[0], c='#172f39', s=70,zorder=7)
        ax.set_xlabel(r'$\theta$ (rad)',fontsize=22)
        ax.set_ylabel(r'$\dot\theta$ (rad/s)',fontsize=22)
        ax.tick_params(labelsize=16)
    axes[0].set(xlim=(THETA_MIN,THETA_MAX),ylim=(-1.8,1.8),title=r'$\theta\in[\gamma-\alpha_{\min},\;\gamma+\alpha_{\max}]$')
    axes[0].annotate('Inward motion can recover\nfrom beyond the rest limits',(-.22,.75),(-.12,1.32),fontsize=17,arrowprops={'arrowstyle':'->', 'lw':1.7}, bbox={'facecolor':'white','edgecolor':'none','pad':3})
    ax=axes[1]
    ax.set(xlim=(-.075,.125),ylim=(-.48,.55),title='Detail near upright')
    ax.set_xticks([-.05, 0, .05, .10])
    ax.plot([0,0],[0,high0[0]],c='#166caa',lw=6,zorder=5,label='Forward capture at section')
    ax.scatter([0,0],[low0[0],high0[0]],s=85,facecolors='white',edgecolors='#166caa',zorder=7)
    ax.annotate(f'{high0[0]:.5f} rad/s',(0,high0[0]),(.025,.35),fontsize=18,color='#166caa',arrowprops={'arrowstyle':'->','color':'#166caa', 'lw':1.7}, bbox={'facecolor':'white','edgecolor':'none','pad':3})
    ax.annotate(f'{low0[0]:.5f} rad/s',(0,low0[0]),(-.04,-.36),fontsize=18,color='#7049a1',arrowprops={'arrowstyle':'->','color':'#7049a1', 'lw':1.7}, bbox={'facecolor':'white','edgecolor':'none','pad':3})
    ax.text(.02,1.02,r'$\theta_L=-0.05002$',fontsize=16, transform=ax.transAxes)
    ax.text(.98,1.02,r'$\theta_R=0.10017$',fontsize=16,ha='right', transform=ax.transAxes)
    ax.annotate('Upright',(0,0),(.055,.22),fontsize=18,arrowprops={'arrowstyle':'->', 'lw':1.7}, bbox={'facecolor':'white','edgecolor':'none','pad':3})
    for ax in axes:
        ax.title.set_fontsize(22)
        ax.set_title(ax.get_title(), fontsize=22, pad=43)
    fig.suptitle('Analytical torque-limited capture region', fontsize=28, y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    extra_handles, extra_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles + [extra_handles[-1]], labels + [extra_labels[-1]],
               loc='lower center', bbox_to_anchor=(.5,.105), ncol=2,
               fontsize=17, frameon=False, columnspacing=2.2, handlelength=2.8)
    fig.text(.5,.065, 'Ideal stance; swing leg held clear; curved boundaries excluded.',
             ha='center', fontsize=17)
    fig.text(.5,.025, r'$u=\tau/(mg\ell)\in[-0.1,0.05]$, $g=9.81$ m/s², $\ell=1$ m. Analytical calculation; no simulations.',
             ha='center', fontsize=16)
    fig.savefig(OUT/'analytical_capture_region.png',dpi=180)
    plt.close(fig)
    assert np.all(lower < upper)
    print(f'Forward capture at theta=0: 0 < omega < {high0[0]:.8f} rad/s')


if __name__ == '__main__':
    main()
