from typing import List
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.pyplot as plt


def plot_task(
    task: List[dict],
    filename: str,
    title: str = None,
    dpi: float = 96,
    hide_last_output: bool = False,
) -> None:
    """
    Saves a task visualization to a PNG file

    Args:
        task: List of dictionaries containing 'input' and 'output' arrays
        filename: Path where the PNG file should be saved
        title: Optional title for the plot
    """
    cmap = ListedColormap(
        [
            "#000",  # black
            "#0074D9",  # blue
            "#FF4136",  # red
            "#2ECC40",  # green
            "#FFDC00",  # yellow
            "#AAAAAA",  # gray
            "#F012BE",  # magenta
            "#FF851B",  # orange
            "#7FDBFF",  # sky
            "#870C25",  # brown
        ]
    )
    norm = Normalize(vmin=0, vmax=9)
    args = {"cmap": cmap, "norm": norm}
    height = 2
    width = len(task)
    figure_size = (width * 3, height * 3)
    figure, axes = plt.subplots(height, width, figsize=figure_size)

    for column, example in enumerate(task):
        axes[0, column].imshow(example["input"], **args)
        if not hide_last_output or column != len(task) - 1:
            axes[1, column].imshow(example["output"], **args)
        axes[0, column].axis("off")
        axes[1, column].axis("off")

    if title is not None:
        figure.suptitle(title, fontsize=20)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(filename, bbox_inches="tight", dpi=dpi)
    plt.close(figure)  # Close the figure to free memory
