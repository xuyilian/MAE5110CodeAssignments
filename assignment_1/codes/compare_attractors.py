"""Draw the basin map and phase portrait side by side from saved basin data.

Run from the repository root: uv run python assignment_1/codes/compare_attractors.py
If basin data are missing, first run: uv run python assignment_1/codes/state_space_basins.py
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from models import rimless_wheel as model
from phase_portrait import plot_phase_portrait
from state_space_basins import plot_basins

OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "figures"


def main():
    data_path = OUTPUT_DIRECTORY / "state_space_basins.npz"
    if not data_path.exists():
        raise FileNotFoundError("Run 'uv run python assignment_1/codes/state_space_basins.py' first")
    with np.load(data_path) as data:
        params = model.generate_params(int(data["number_of_spokes"]))
        params.update({key: data[key].item() for key in params if key in data})
        figure, axes = plt.subplots(1, 2, figsize=(16, 9))
        figure.subplots_adjust(left=0.07, right=0.98, bottom=0.32, top=0.89, wspace=0.28)
        basin_handles = plot_basins(
            data["initial_angles"], data["initial_velocities"],
            data["classifications"], data["steady_post_impact_velocities"], params,
            axis=axes[0],
        )
        phase_handles = plot_phase_portrait(params, axis=axes[1])

    for letter, axis, handles in zip("ab", axes, [basin_handles, phase_handles]):
        axis.set_title(f"({letter}) " + axis.get_title(), fontsize=15, pad=12)
        axis.xaxis.label.set_size(14)
        axis.yaxis.label.set_size(14)
        axis.tick_params(labelsize=12)
        axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                    ncol=2, frameon=False, fontsize=12, columnspacing=1.1)
    output_path = OUTPUT_DIRECTORY / "basins_and_phase_portrait.png"
    figure.savefig(output_path, dpi=250)
    print(f"Saved {output_path}")
    return figure, output_path


if __name__ == "__main__":
    main()
    plt.show()
