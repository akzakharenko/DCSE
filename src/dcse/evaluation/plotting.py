import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde


def scatter_plot(
    true,
    pred,
    label,
    save_path
):

    plt.figure(
        figsize=(4, 3),
        dpi=200
    )

    xy = np.vstack([true, pred])

    z = gaussian_kde(xy)(xy)

    idx = z.argsort()

    true = true[idx]
    pred = pred[idx]
    z = z[idx]

    sc = plt.scatter(
        true,
        pred,
        c=z,
        s=8,
        cmap="viridis"
    )

    lims = [
        min(true.min(), pred.min()),
        max(true.max(), pred.max())
    ]

    plt.plot(
        lims,
        lims,
        linestyle="--",
        color="orange",
        linewidth=1.2
    )

    plt.xlabel(
        "True",
        fontsize=10
    )

    plt.ylabel(
        "Predicted",
        fontsize=10
    )

    plt.title(
        label,
        fontsize=10
    )

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.grid(
        True,
        linewidth=0.4
    )

    cbar = plt.colorbar(sc)

    cbar.set_label(
        "Density",
        fontsize=10
    )

    cbar.ax.tick_params(
        labelsize=10
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0.05
    )

    plt.close()