import logging

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

fig, ax = plt.subplots(figsize=(6, 7))
ax.set_xlim(0, 6)
ax.set_ylim(0, 8)
ax.axis("off")


def box(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    facecolor: str,
    edgecolor: str = "black",
    lw: float = 1.5,
    zorder: int = 2,
    linestyle: str = "-",
) -> None:
    """Draw a rounded rectangle patch on the given axes."""
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.05",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=zorder,
        linestyle=linestyle,
    )
    ax.add_patch(rect)


def arrow(
    ax: Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "black",
    lw: float = 1.5,
    zorder: int = 3,
) -> None:
    """Draw a straight annotated arrow between two points on the given axes."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": lw,
            "mutation_scale": 14,
        },
        zorder=zorder,
    )


def label(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    fontsize: int = 11,
    color: str = "black",
    ha: str = "center",
    va: str = "center",
    style: str = "normal",
    weight: str = "normal",
) -> None:
    """Place a text label at the given coordinates on the axes."""
    ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        color=color,
        ha=ha,
        va=va,
        style=style,
        fontweight=weight,
        zorder=5,
    )


FROZEN_BLUE = "#BDD7EE"
TRAINABLE_ORANGE = "#FCE4D6"
EDGE_FROZEN = "#2166AC"
EDGE_TRAIN = "#C55A11"
ARROW_COL = "#333333"

cx = 3.0
W_main = 2.2
H_main = 1.6

W_lora = 0.85
H_A = 0.65
H_B = 0.65

y_input = 0.7
y_x_node = 1.5
y_w = 3.8
y_add = 5.9
y_output = 7.0

lora_x = cx + 1.85
Y_A = 2.5
Y_B = 5.0

box(ax, cx, y_w, W_main, H_main, facecolor=FROZEN_BLUE, edgecolor=EDGE_FROZEN, lw=2.0)

label(
    ax,
    cx,
    y_w + 0.28,
    r"Pretrained Weights",
    fontsize=10,
    weight="bold",
    color=EDGE_FROZEN,
)
label(
    ax,
    cx,
    y_w - 0.22,
    r"$W \in \mathbb{R}^{d \times d}$",
    fontsize=11,
    color=EDGE_FROZEN,
)

box(
    ax,
    lora_x,
    Y_A,
    W_lora,
    H_A,
    facecolor=TRAINABLE_ORANGE,
    edgecolor=EDGE_TRAIN,
    lw=2.0,
)

label(ax, lora_x, Y_A + 0.05, r"$A$", fontsize=13, weight="bold", color=EDGE_TRAIN)
label(
    ax,
    lora_x,
    Y_A - 0.24,
    r"$\mathcal{N}(0,\,\sigma^2)$",
    fontsize=8.5,
    color="#555555",
)

box(
    ax,
    lora_x,
    Y_B,
    W_lora,
    H_A,
    facecolor=TRAINABLE_ORANGE,
    edgecolor=EDGE_TRAIN,
    lw=2.0,
)

label(ax, lora_x, Y_B + 0.05, r"$B$", fontsize=13, weight="bold", color=EDGE_TRAIN)
label(ax, lora_x, Y_B - 0.24, r"$= 0$", fontsize=9, color="#555555")

label(ax, lora_x + 0.58, (y_x_node + Y_A) / 2, r"$d$", fontsize=10, color="#666666")
label(
    ax,
    lora_x + 0.58,
    (Y_A + Y_B) / 2,
    r"$r$",
    fontsize=10,
    color="#666666",
    style="italic",
)
label(ax, lora_x + 0.58, (Y_B + y_add) / 2, r"$d$", fontsize=10, color="#666666")

input_circle = plt.Circle(
    (cx, y_x_node), 0.22, color="#EEEEEE", ec="black", lw=1.5, zorder=2
)
ax.add_patch(input_circle)
label(ax, cx, y_x_node, r"$x$", fontsize=12)

add_circle = plt.Circle(
    (cx, y_add), 0.28, color="#E2EFDA", ec="#375623", lw=1.8, zorder=2
)
ax.add_patch(add_circle)
label(ax, cx, y_add, r"$+$", fontsize=16, color="#375623", weight="bold")

label(ax, cx, y_output, r"$h = Wx + BAx$", fontsize=11)

arrow(ax, cx, y_x_node + 0.22, cx, y_w - H_main / 2, color=ARROW_COL)
arrow(ax, cx, y_w + H_main / 2, cx, y_add - 0.28, color=ARROW_COL)
arrow(ax, cx, y_add + 0.28, cx, y_output - 0.18, color=ARROW_COL)

ax.annotate(
    "",
    xy=(lora_x, Y_A - H_A / 2),
    xytext=(cx, y_x_node),
    arrowprops={
        "arrowstyle": "-|>",
        "color": EDGE_TRAIN,
        "lw": 1.8,
        "connectionstyle": "arc3,rad=-0.25",
        "mutation_scale": 14,
    },
    zorder=3,
)

arrow(ax, lora_x, Y_A + H_A / 2, lora_x, Y_B - H_A / 2, color=EDGE_TRAIN, lw=1.8)

ax.annotate(
    "",
    xy=(cx + 0.28, y_add),
    xytext=(lora_x, Y_B + H_A / 2),
    arrowprops={
        "arrowstyle": "-|>",
        "color": EDGE_TRAIN,
        "lw": 1.8,
        "connectionstyle": "arc3,rad=0.25",
        "mutation_scale": 14,
    },
    zorder=3,
)

arrow(ax, cx, y_input, cx, y_x_node - 0.23, color=ARROW_COL)
label(ax, cx, y_input - 0.22, r"input $x$", fontsize=10, color="#444444")

frozen_patch = mpatches.Patch(
    facecolor=FROZEN_BLUE, edgecolor=EDGE_FROZEN, lw=1.5, label="Frozen (not updated)"
)
trainable_patch = mpatches.Patch(
    facecolor=TRAINABLE_ORANGE, edgecolor=EDGE_TRAIN, lw=1.5, label="Trainable (LoRA)"
)
ax.legend(
    handles=[frozen_patch, trainable_patch],
    loc="lower left",
    fontsize=9,
    bbox_to_anchor=(0.0, 0.0),
    framealpha=0.85,
)

ax.set_title(
    "LoRA Reparametrisation of a Weight Matrix", fontsize=12, fontweight="bold", pad=10
)

fig.tight_layout()
fig.savefig("outputs/fig_lora_architecture.pdf", bbox_inches="tight")
fig.savefig("outputs/fig_lora_architecture.png", bbox_inches="tight")
plt.close(fig)
logger.info("LoRA architecture diagram saved.")
