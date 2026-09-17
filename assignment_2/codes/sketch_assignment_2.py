"""Generate analytical sketches, independent of trajectory integration."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np

OUTPUT = Path(__file__).resolve().parents[1] / 'figures'
G, L, GAMMA = 9.81, 1.0, 0.06
ANGLES = [np.pi / 8, np.pi / 7]
COLORS = ['#c06419', '#7851a9']


def draw_pose(ax, theta, alpha, title, note):
    hub = L * np.array([np.sin(theta), np.cos(theta)])
    swing = hub - L * np.array([np.sin(theta - 2 * alpha), np.cos(theta - 2 * alpha)])
    x = np.linspace(-0.65, 1.2, 100)
    ax.fill_between(x, -np.tan(GAMMA) * x, -0.3, color='#ede7dd')
    ax.plot(x, -np.tan(GAMMA) * x, color='#80684e')
    ax.plot([0, 0], [0, 1.22], ':', color='gray')
    ax.plot([0, hub[0]], [0, hub[1]], '-o', color='#23699b', lw=3)
    ax.plot([hub[0], swing[0]], [hub[1], swing[1]], '--o', color='#c06419', lw=2)
    ax.scatter(*hub, s=130, c='#23699b', zorder=5)
    # Draw force independently of its text so the arrow stays vertical.
    ax.annotate('', hub + [0, -0.43], hub,
                arrowprops={'arrowstyle': '->', 'color': '#ad2831', 'lw': 1.7}, zorder=7)
    ax.text(hub[0] - .07, hub[1] - .39, r'$mg$', color='#ad2831',
            ha='right', va='center', bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
    ax.text(hub[0] + .06, hub[1] + .06, r'$m$')
    ax.text(hub[0]/2 - .13, hub[1]/2, r'$\ell$')
    if abs(theta) > .01:
        ends = sorted([90 - np.degrees(theta), 90])
        ax.add_patch(Arc((0, 0), .48, .48, theta1=ends[0], theta2=ends[1], color='#23699b'))
        mid = np.pi/2 - theta/2
        # The geometric arc is a magnitude; theta is negative in snapshot D.
        label = r'$-\theta$' if theta < 0 else r'$\theta$'
        ax.text(.34*np.cos(mid), .34*np.sin(mid), label,
                ha='center', va='center', color='#23699b',
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
    angle_start = 270 - np.degrees(theta)
    ax.add_patch(Arc(hub, .46, .46, theta1=angle_start,
                     theta2=angle_start + 2*np.degrees(alpha), color='#555555'))
    bisector = np.radians(angle_start) + alpha
    label_position = hub + .23*np.array([np.cos(bisector), np.sin(bisector)])
    # Offset a callout to avoid the vertical gravity arrow near touchdown.
    text_position = label_position + [.27, .04]
    ax.annotate(r'$2\alpha$', xy=label_position, xytext=text_position,
                ha='left', va='center', color='#555555',
                arrowprops={'arrowstyle': '-', 'color': '#777777'},
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
    ax.set(title=title, xlim=(-.65, 1.2), ylim=(-.25, 1.3), xlabel='x (m)', ylabel='y (m)')
    ax.text(.02, .02, note, transform=ax.transAxes, fontsize=9, va='bottom')
    ax.set_aspect('equal')


def plot_snapshots():
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout='constrained')
    draw_pose(axes[0,0], 0, ANGLES[0], 'A: mid-stance', r'$\theta=0,\quad\dot\theta=1.5$ rad/s')
    for ax, a, label in zip([axes[0,1], axes[1,0]], ANGLES, ['B', 'C']):
        td = a + GAMMA
        speed = np.sqrt(1.5**2 + 2*G/L*(1-np.cos(td)))
        draw_pose(ax, td, a, f'{label}: just before touchdown', rf'$\alpha={a:.4f}$ rad; $\theta^-={td:.4f}$ rad' + '\n' + rf'$\dot{{\theta}}^-={speed:.3f}$ rad/s')
    draw_pose(axes[1,1], -.25, ANGLES[0], 'D: insufficient energy to pass upright', r'$\theta=-0.25$ rad; $\dot\theta=0$ rad/s'+'\nGravity accelerates the walker backward.')
    fig.suptitle(r'Walker geometry on a downhill slope: $\gamma=0.06$ rad' + '\n' + r'Blue: stance leg; dashed orange: swing leg; red: gravity; $\tau=0$', fontsize=13)
    fig.savefig(OUTPUT/'sketch_snapshots.png', dpi=180)
    plt.close(fig)


def plot_phase_space():
    fig, ax = plt.subplots(figsize=(11, 7), layout='constrained')
    theta = np.linspace(-.65, .65, 43)
    omega = np.linspace(-2.7, 2.7, 43)
    tt, ww = np.meshgrid(theta, omega)
    ax.streamplot(theta, omega, ww, G/L*np.sin(tt), color='#dadada', density=.8, arrowsize=.8)
    sep = np.sqrt(2*G/L*(1-np.cos(theta)))
    ax.plot(theta, sep, '--', color='#777777', label='Upright-energy separatrix')
    ax.plot(theta, -sep, '--', color='#777777')
    ax.axvline(0, color='#23699b', lw=1, label='Candidate section: theta = 0, omega > 0')
    ax.scatter(0, 0, c='black', s=35)
    ax.annotate('Upright saddle', (0,0), (.06,-.35))
    ax.scatter(0,1.5,c='#23699b', zorder=6)
    ax.annotate('A', (0,1.5), (-.04,1.7), weight='bold')
    for a, c, label in zip(ANGLES, COLORS, ['B','C']):
        td, reset = GAMMA+a, GAMMA-a
        pre = np.sqrt(1.5**2 + 2*G/L*(1-np.cos(td)))
        post = np.cos(2*a)*pre
        ax.axvline(td, c=c, lw=1.8, label=f'Touchdown guard: alpha = {a:.4f}')
        ts = np.linspace(0,td,100)
        ax.plot(ts,np.sqrt(1.5**2+2*G/L*(1-np.cos(ts))), c=c, lw=2)
        ax.scatter([td,reset],[pre,post],c=c,zorder=5)
        ax.annotate(label+'−', (td,pre), (td+.015,pre+.10), color=c, weight='bold')
        ax.annotate(label+'+', (reset,post), (reset-.06,post+.13), color=c, weight='bold')
        ax.annotate('', (reset,post), (td,pre), arrowprops={'arrowstyle':'->','color':c,'linestyle':'--','connectionstyle':f'arc3,rad={.15 if label=="B" else -.12}'})
    ts=np.linspace(-.60,-.25,120)
    speed=np.sqrt(np.maximum(0,2*G/L*(np.cos(-.25)-np.cos(ts))))
    ax.plot(ts,speed,c='#ad2831',lw=2)
    ax.plot(ts,-speed,c='#ad2831',lw=2,label='D: approach, turn, and fall backward')
    ax.scatter(-.25,0,c='#ad2831',zorder=5)
    ax.annotate('D: turning point',(-.25,0),(-.60,-.40),arrowprops={'arrowstyle':'->'},color='#ad2831')
    ax.annotate('',(-.47,-np.sqrt(2*G/L*(np.cos(-.25)-np.cos(-.47)))),(-.40,-np.sqrt(2*G/L*(np.cos(-.25)-np.cos(-.40)))),arrowprops={'arrowstyle':'->','color':'#ad2831','lw':2})
    ax.set(xlim=(-.65,.65),ylim=(-2.7,2.7),xlabel=r'$\theta$ (rad, clockwise from upward vertical)',ylabel=r'$\dot\theta$ (rad/s)',title='Uncontrolled stance phase portrait and impact resets\nB and C are alternative choices from the same mid-stance state A')
    ax.title.set_fontsize(17)
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.tick_params(axis='both', labelsize=13)
    for label in ax.texts:
        label.set_fontsize(15)
    ax.legend(loc='lower right', fontsize=11, framealpha=.95)
    fig.savefig(OUTPUT/'sketch_phase_space.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    OUTPUT.mkdir(parents=True,exist_ok=True)
    plot_snapshots()
    plot_phase_space()
