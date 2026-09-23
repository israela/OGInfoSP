import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# Common constants
SETS = [
    (r"$\{1\}$", {1}, 1.0), (r"$\{2\}$", {2}, 1.0), (r"$\{3\}$", {3}, 1.0),
    (r"$\{1,2\}$", {1,2}, 0.5), (r"$\{1,3\}$", {1,3}, 0.5), (r"$\{2,3\}$", {2,3}, 0.5),
]

COLORS = {
    r"$\{1\}$": "#2060B0", r"$\{2\}$": "#C03030", r"$\{3\}$": "#F0A020", 
    r"$\{1,2\}$": "#1A8A3A", r"$\{1,3\}$": "#D07020", r"$\{2,3\}$": "#A020F0"
}


def create_envelope_plot(probs, xlim, ylim, show_D_label, output_filename, label_positions, sets, envelope, show_legend=True, legend_loc="upper right", with_hats=False, alpha=0.1, mu_max=5.2):
    """Create and save an envelope illustration.

    Args:
        show_D_label: Whether to show the D=0 region and label.
        label_positions: Mapping of region keys to positions or custom labels and positions.
        sets: Sequence of (label, class members, weight) tuples.
        envelope: Two labels from sets that form the envelope.
        with_hats: Whether to use hatted mathematical notation.
    """
    ell = r"\hat{\ell}" if with_hats else r"\ell"
    C_mu = r"\hat{C}^{\mu}" if with_hats else r"C^{\mu}"
    D_mu = r"\hat{D}^{\mu}" if with_hats else r"D^{\mu}"

    lines = {}
    for name, members, w in sets:
        p = sum(probs[k] for k in members)
        lines[name] = {"intercept": w * p, "slope": p - (1 - alpha), "w": w, "p": p}

    L1, L2 = lines[envelope[0]], lines[envelope[1]]
    mu_break = (L1["intercept"] - L2["intercept"]) / (L2["slope"] - L1["slope"])
    y_break = L1["intercept"] + L1["slope"] * mu_break
    mu_zero = -L2["intercept"] / L2["slope"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    mu = np.linspace(0, mu_max, 600)

    # Glow behind envelope
    env1_mu = np.linspace(0, mu_break, 200)
    env2_mu = np.linspace(mu_break, mu_max, 200)
    env1_y = L1["intercept"] + L1["slope"] * env1_mu
    env2_y = L2["intercept"] + L2["slope"] * env2_mu
    ax.plot(env1_mu, env1_y, color="#FFD966", linewidth=10, alpha=0.55, solid_capstyle="round", zorder=1)
    ax.plot(env2_mu, env2_y, color="#FFD966", linewidth=10, alpha=0.55, solid_capstyle="round", zorder=1)

    # Highlight D=0 region (x-axis to the right of zero crossing)
    if show_D_label and mu_zero > 0:
        d0_mu = np.linspace(mu_zero, mu_max, 200)
        d0_y = np.zeros_like(d0_mu)
        ax.plot(d0_mu, d0_y, color="#FF6B6B", linewidth=6, alpha=0.55, solid_capstyle="round", zorder=1)

    # Draw all lines as full linear functions
    for name, L in lines.items():
        vals = L["intercept"] + L["slope"] * mu
        on_env = name in envelope
        ax.plot(mu, vals, color=COLORS[name], linewidth=1.8 if on_env else 1.5,
                alpha=0.9 if on_env else 0.75, linestyle="-" if on_env else "--",
                label=fr"${ell}_{{x,{name[1:-1]}}}(\mu)$", zorder=3)

    # Highlighted markers — diamond for breakpoint, circle for zero crossing
    ax.plot(mu_break, y_break, "D", color="#9B2694", markersize=8, zorder=6, markeredgecolor="white", markeredgewidth=0.8)
    if mu_zero > 0:  # Only show zero crossing if it's in the positive range
        ax.plot(mu_zero, 0, "o", color="#E8840C", markersize=8, zorder=6, markeredgecolor="white", markeredgewidth=0.8)

    # Region labels
    for key, color, default_text in [('C1', COLORS[envelope[0]], fr"${C_mu}(x)\!=\!{envelope[0][1:-1]}$"),
                                      ('C12', COLORS[envelope[1]], fr"${C_mu}(x)\!=\!{envelope[1][1:-1]}$"),
                                      ('C2', COLORS[envelope[1]], fr"${C_mu}(x)\!=\!{envelope[1][1:-1]}$")]:
        if key in label_positions:
            val = label_positions[key]
            if isinstance(val[0], str):  # (label_text, (mu, y))
                text, (lx, ly) = val
            else:  # (mu, y) — use default text
                lx, ly = val
                text = default_text
            ax.text(lx, ly, text, fontsize=14, color=color, fontweight="bold")
    
    if show_D_label and 'D0' in label_positions:
        val = label_positions['D0']
        if isinstance(val[0], str):
            text, (lx, ly) = val
        else:
            lx, ly = val
            text = fr"${D_mu}(x)\!=\!0$"
        ax.text(lx, ly, text, fontsize=14, color="0.4", fontweight="bold")

    ax.axvline(mu_break, color="grey", linestyle=":", linewidth=0.6, alpha=0.4, zorder=0)
    ax.set(xlim=xlim, ylim=ylim, xlabel=r"$\mu$")
    ax.set_ylabel(fr"${ell}_{{x,C}}(\mu)$", fontsize=16, rotation=0, labelpad=22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="black", linewidth=0.6, zorder=0.5)
    ax.axvline(0, color="black", linewidth=0.6, zorder=0.5)

    # Custom legend
    if show_legend:
        legend_elements = [
            Line2D([0], [0], color="#FFD966", linewidth=8, alpha=0.55, label=r"$\mathcal{U}_x(\mu)$"),
            Line2D([0], [0], color="#FF6B6B", linewidth=5, alpha=0.55, label=fr"${D_mu}(x) = 0$"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#9B2694", markersize=8, 
                   markeredgecolor="white", markeredgewidth=0.8, label="breakpoint"),
        ]
        if mu_zero > 0:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#E8840C", markersize=8, 
                       markeredgecolor="white", markeredgewidth=0.8, label="zero crossing")
            )
        legend_elements += [ax.get_legend_handles_labels()[0][i] for i in range(len(sets))]
        ax.legend(handles=legend_elements, loc=legend_loc, fontsize=13, framealpha=0.92)

    fig.savefig(output_filename, dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()


def create_legend_figure(output_filename, sets, envelope, with_hats=False):
    """Create and save a standalone envelope legend.

    Args:
        sets: Sequence of (label, class members, weight) tuples.
        envelope: Two labels from sets that form the envelope.
        with_hats: Whether to use hatted mathematical notation.
    """
    ell = r"\hat{\ell}" if with_hats else r"\ell"
    D_mu = r"\hat{D}^{\mu}" if with_hats else r"D^{\mu}"
    legend_elements = [
        Line2D([0], [0], color="#FFD966", linewidth=8, alpha=0.55, label=r"$\mathcal{U}_x(\mu)$"),
        Line2D([0], [0], color="#FF6B6B", linewidth=8, alpha=0.55, label=fr"${D_mu}(x) = 0$"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#9B2694", markersize=8, 
               markeredgecolor="white", markeredgewidth=0.8, label="breakpoint"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E8840C", markersize=8, 
               markeredgecolor="white", markeredgewidth=0.8, label="zero crossing"),
    ]
    for name, members, w in sets:
        on_env = name in envelope
        legend_elements.append(
            Line2D([0], [0], color=COLORS[name], linewidth=1.8 if on_env else 1.5,
                   alpha=0.9 if on_env else 0.75, 
                   linestyle="-" if on_env else "--",
                   label=fr"${ell}_{{x,{name[1:-1]}}}(\mu)$"))

    fig = plt.figure()
    legend = fig.legend(handles=legend_elements, loc='center', fontsize=13, framealpha=0.92)
    for txt in legend.get_texts()[-len(sets):]:
        txt.set_fontsize(15)
    fig.canvas.draw()
    
    renderer = fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer)
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    w, h = bbox_inches.width, bbox_inches.height
    fig.set_size_inches(w + 0.04, h + 0.04)
    legend.set_bbox_to_anchor((0.5, 0.5), transform=fig.transFigure)
    
    fig.savefig(output_filename, dpi=400)
    plt.show()
    plt.close()


for with_hats in [True, False] if __name__ == '__main__' else []:
    suffix = "_with_hats" if with_hats else ""

    # Generate illustration with high probabilities
    probs1 = {1: 0.50, 2: 0.30, 3: 0.20}
    label_pos1 = {
        'C1': (0.15, 0.46),
        'C12': (1.6, 0.26),
        'D0': (4.17, 0.018)
    }
    create_envelope_plot(probs1, xlim=(-0.05, 5.2), ylim=(-0.15, 0.56), show_D_label=True,
                         output_filename=f"output/figures/upper_envelope_low_prob{suffix}.jpg",
                         label_positions=label_pos1, sets=SETS,
                         envelope=[r"$\{1\}$", r"$\{1,2\}$"], show_legend=False,
                         with_hats=with_hats)

    # Generate illustration with low probabilities
    probs2 = {1: 0.70, 2: 0.25, 3: 0.05}
    label_pos2 = {
        'C1': (0.20, 0.69),
        'C12': (1.6, 0.52)
    }
    create_envelope_plot(probs2, xlim=(-0.05, 5.2), ylim=(-0.05, 0.75), show_D_label=False,
                         output_filename=f"output/figures/upper_envelope_high_prob{suffix}.jpg",
                         label_positions=label_pos2, sets=SETS,
                         envelope=[r"$\{1\}$", r"$\{1,2\}$"], show_legend=False,
                         with_hats=with_hats)

    # Generate standalone legend
    create_legend_figure(f"output/figures/upper_envelope_legend{suffix}.jpg",
                         sets=SETS, envelope=[r"$\{1\}$", r"$\{1,2\}$"],
                         with_hats=with_hats)

    # Generate the assumption counterexample figure: p1=0.4, p2=0.35, p3=0.25 with I={{1},{2},{3},{2,3}}
    probs3 = {1: 0.40, 2: 0.35, 3: 0.25}
    sets3 = [
        (r"$\{1\}$", {1}, 1.0), (r"$\{2\}$", {2}, 1.0), (r"$\{3\}$", {3}, 1.0),
        (r"$\{2,3\}$", {2, 3}, 0.5),
    ]
    label_pos3 = {
        'C1': (0.25, 0.3),
        'C2': (0.75, 0.1),
        'D0': (1.8, 0.018),
    }
    create_envelope_plot(probs3, xlim=(-0.05, 2.7), ylim=(-0.15, 0.45), show_D_label=True,
                         output_filename=f"output/figures/upper_envelope_subset_23{suffix}.jpg",
                         label_positions=label_pos3, show_legend=False,
                         sets=sets3, envelope=[r"$\{1\}$", r"$\{2,3\}$"],
                         with_hats=with_hats)

    # Generate legend for the third figure
    create_legend_figure(f"output/figures/upper_envelope_subset_23_legend{suffix}.jpg",
                         sets=sets3, envelope=[r"$\{1\}$", r"$\{2,3\}$"],
                         with_hats=with_hats)