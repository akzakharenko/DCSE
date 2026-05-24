import os
import matplotlib.pyplot as plt


def save_history_plots(
    history,
    out_dir="train_history"
):

    os.makedirs(out_dir, exist_ok=True)

    for key, values in history.history.items():

        plt.figure(figsize=(4, 4))

        plt.plot(values)

        plt.title(key)

        plt.xlabel("Epoch")
        plt.ylabel(key)

        plt.tight_layout()

        plt.savefig(
            os.path.join(out_dir, f"{key}.png")
        )

        plt.close()