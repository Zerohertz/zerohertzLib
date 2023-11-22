from matplotlib import pyplot as plt


def _save(title: str, dpi: int) -> None:
    title = title.lower().replace(" ", "_").replace("/", "-")
    plt.savefig(
        f"{title}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close("all")
