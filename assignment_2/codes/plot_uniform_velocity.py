"""Overlay the uniform velocity grid on the fine state-action solution."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from validate_uniform_velocity import table, execute
from build_lookup_policy import CAPTURE, LIMIT, INF, OUT


def draw_figure(omega, alpha, cost):
    nodes, actions = table(3)
    degrees = np.degrees(actions)
    switch = LIMIT/4
    fig = plt.figure(figsize=(14, 12), layout='constrained')
    gs = fig.add_gridspec(3, 2, width_ratios=[1, .035], height_ratios=[1.4, 1, .65])
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[i, 0], sharex=axes[0]) for i in [1, 2]]
    ax = axes[0]
    colors = plt.get_cmap('viridis', 3).copy()
    colors.set_bad('#eeeeee')
    mesh = ax.pcolormesh(omega, np.degrees(alpha),
                         np.ma.masked_where(cost.T >= INF, cost.T),
                         cmap=colors, shading='nearest', vmin=.5, vmax=3.5)
    ax.plot([CAPTURE, switch, switch, LIMIT],
            [degrees[0], degrees[0], degrees[1], degrees[2]],
            color='white', lw=3, zorder=6)
    ax.scatter(nodes, degrees, s=140, c='#e97032', edgecolors='white',
               linewidths=1.8, zorder=8)
    ax.set_ylim(22.25, 26)
    ax.set_yticks(degrees[:2], ['22.5', '25.7143'])
    ax.set_ylabel(r'Candidate $\alpha$ (degrees)', fontsize=19)
    ax.set_title('Fine-grid solution with two equal velocity intervals', fontsize=23, pad=62)
    for j, (left, right) in enumerate(zip(nodes[:-1], nodes[1:]), 1):
        ax.annotate('', (right, 1.035), (left, 1.035),
                    xycoords=ax.get_xaxis_transform(),
                    arrowprops={'arrowstyle': '<->', 'lw': 1.6}, annotation_clip=False)
        ax.text((left+right)/2, 1.06, f'Grid interval {j}: '+r'$\Delta\omega=2.21472$',
                transform=ax.get_xaxis_transform(), ha='center', fontsize=17)
    cb = fig.colorbar(mesh, cax=fig.add_subplot(gs[0, 1]), ticks=[1, 2, 3])
    cb.set_label('Footstrikes, then follow fine policy', fontsize=16)
    cb.ax.tick_params(labelsize=15)
    ax = axes[1]
    ax.plot([0, CAPTURE], [degrees[0]]*2, ':', color='#ad572c', lw=2)
    ax.plot([CAPTURE, switch, switch, LIMIT],
            [degrees[0], degrees[0], degrees[1], degrees[2]],
            color='#244b70', lw=3)
    ax.scatter(nodes, degrees, s=140, c='#e97032', edgecolors='white', zorder=5)
    ax.annotate(r'Node 0: $\pi/8$', (0, degrees[0]), xytext=(.10, 23.4), fontsize=17,
                arrowprops={'arrowstyle': '-', 'color': '#ad572c'})
    for i in [1, 2]:
        ax.text(nodes[i] + (-.07 if i == 2 else .07), 26.15,
                f'Node {i}: $\\pi/7$', ha='right' if i == 2 else 'left', fontsize=17)
    ax.axvline(switch, color='#567187', ls='--', lw=1.3)
    ax.text(switch+.10, 24, 'Action switch\n1.10736 rad/s', fontsize=16)
    ax.set_ylim(21.9, 27)
    ax.set_yticks(degrees[:2], ['22.5', '25.7143'])
    ax.set_ylabel(r'Selected $\alpha$ (degrees)', fontsize=19)
    ax.set_title('Stored node actions and executed walking policy', fontsize=20, pad=12)
    starts = np.unique(np.r_[np.linspace(0, LIMIT, 10001), CAPTURE])
    counts = execute(starts, 3)
    axes[2].step(starts, counts, where='post', color='#75468b', lw=2.8)
    axes[2].set_ylim(-.15, 3.35)
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_ylabel('Footstrikes\nto standing RoA', fontsize=19)
    axes[2].set_xlabel(r'Section velocity $\omega_k$ (rad/s)', fontsize=20)
    for ax in axes:
        ax.axvspan(0, CAPTURE, color='#c9e7df', zorder=3 if ax is axes[0] else 0, alpha=.8)
        for node in nodes:
            ax.axvline(node, color='#252525', lw=1.5, zorder=4)
        ax.set_xlim(-.05, LIMIT+.05)
        ax.set_xticks(nodes, ['0', '2.21472', '4.42945'])
        ax.tick_params(labelsize=16)
    fig.supxlabel('Solid vertical lines: uniform grid nodes. Orange dots: stored actions.\n'
                  'Green: standing control overrides lookup. Gray in the fine map: unresolved action.',
                  fontsize=15)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/'lookup_state_action.png', dpi=180)
    plt.close(fig)


def main():
    with np.load(OUT/'lookup_policy.npz') as data:
        draw_figure(data['omega'], data['alpha'], data['cost'])


if __name__ == '__main__':
    main()
