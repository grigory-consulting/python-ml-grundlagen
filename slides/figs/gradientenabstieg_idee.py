"""Gradientenabstieg als Idee: Kostenkurve, Lernschritte, Gradient und Gegenrichtung am Startpunkt."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ohne Argument landet das Bild neben dem Skript
out = sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).with_suffix(".png")

INK, BLUE, RED, GOLD = "#0f172a", "#2563eb", "#dc2626", "#fde68a"
theta_min, theta_start, eta = 5.9, 2.8, 0.125


def cost(t):
    return (t - theta_min) ** 2 * 1.4 + 4.0


def grad(t):
    return 2 * 1.4 * (t - theta_min)


# Lernschritte: theta <- theta - eta * Gradient
thetas = [theta_start]
for _ in range(9):
    thetas.append(thetas[-1] - eta / 1.4 * grad(thetas[-1]))

fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=145)
t = np.linspace(0, 10, 400)
ax.plot(t, cost(t), color=INK, lw=3, zorder=2)

c0 = cost(theta_start)

# Gradient und Schritt als Richtungen auf der Parameterachse
y_arrow = 0.8
ax.annotate("", xy=(theta_start - 1.5, y_arrow), xytext=(theta_start, y_arrow),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=3, mutation_scale=22,
                            shrinkA=0, shrinkB=0), zorder=5)
ax.text(theta_start - 0.15, 3.1, "Gradient zeigt\nbergauf", color=RED, fontsize=12.5,
        fontweight="bold", va="center", ha="right")
ax.annotate("", xy=(theta_start + 1.5, y_arrow), xytext=(theta_start, y_arrow),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=3, mutation_scale=22,
                            shrinkA=0, shrinkB=0), zorder=5)
ax.text(theta_start + 0.15, 3.1, "Schritt in die\nGegenrichtung", color=BLUE, fontsize=12.5,
        fontweight="bold", va="center", ha="left")

# Lernschritte als Bögen
for a, b in zip(thetas[:-1], thetas[1:]):
    ax.annotate("", xy=(b, cost(b)), xytext=(a, cost(a)),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.5,
                                connectionstyle="arc3,rad=-0.45", shrinkA=5, shrinkB=5),
                zorder=3)
ax.text(4.4, 15.2, "Lernschritt", fontsize=12.5, style="italic", color=INK)

ax.scatter(thetas, [cost(v) for v in thetas], s=110, color=BLUE, ec=INK, lw=1, zorder=6)
ax.scatter([theta_min], [cost(theta_min)], s=140, color=GOLD, ec=INK, lw=1, zorder=6)
ax.text(theta_min + 0.18, cost(theta_min) + 1.0, "Minimum", fontsize=15, fontweight="bold")

ax.plot([theta_start] * 2, [0, c0], color="#334155", lw=1.8, ls="--", zorder=1)
ax.plot([theta_min] * 2, [0, cost(theta_min)], color="#334155", lw=1.2, zorder=1)

ax.set_xlim(0.3, 10.5)
ax.set_ylim(0, 24)
ax.set_xticks([theta_start, theta_min])
ax.set_xticklabels(["zufälliger Startwert", "bester Wert"], fontsize=12.5, style="italic")
ax.tick_params(axis="x", length=0, pad=8)
ax.set_yticks([])
ax.set_ylabel("Kosten", fontsize=13)
ax.set_xlabel("Parameter θ", fontsize=13, loc="right", labelpad=-14)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_linewidth(2)
fig.tight_layout()
fig.savefig(out)
