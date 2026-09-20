"""Pfade von Batch-, Mini-Batch- und stochastischem Gradientenabstieg im Parameterraum (b, m)."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ohne Argument landet das Bild neben dem Skript
out = sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).with_suffix(".png")

# Daten wie in den anderen GD-Grafiken: y = 4 + 3x + Rauschen
rng = np.random.default_rng(1)
n = 100
x = 2 * rng.random(n)
y = 4 + 3 * x + rng.normal(size=n)
X = np.c_[np.ones(n), x]

# exaktes Minimum über die Normalengleichung
theta_best = np.linalg.solve(X.T @ X, X.T @ y)

theta_start = np.array([0.0, 0.0])
n_epochs = 30


def gradient(theta, idx):
    Xb, yb = X[idx], y[idx]
    return 2 / len(idx) * Xb.T @ (Xb @ theta - yb)


def schedule(t, t0, t1):
    # abnehmende Lernrate, damit der Pfad am Minimum zur Ruhe kommt
    return t0 / (t + t1)


# Batch: alle Punkte, feste Lernrate
theta = theta_start.copy()
path_batch = [theta.copy()]
for _ in range(1000):
    theta = theta - 0.1 * gradient(theta, np.arange(n))
    path_batch.append(theta.copy())

# SGD: ein Punkt je Schritt
theta = theta_start.copy()
path_sgd = [theta.copy()]
t = 0
for epoch in range(n_epochs):
    for i in rng.permutation(n):
        theta = theta - schedule(t, 5, 50) * gradient(theta, np.array([i]))
        path_sgd.append(theta.copy())
        t += 1

# Mini-Batch: 8 Punkte je Schritt
theta = theta_start.copy()
path_mini = [theta.copy()]
t = 0
for epoch in range(2 * n_epochs):
    perm = rng.permutation(n)
    for k in range(0, n, 8):
        theta = theta - schedule(t, 20, 200) * gradient(theta, perm[k:k + 8])
        path_mini.append(theta.copy())
        t += 1

path_batch, path_sgd, path_mini = map(np.array, (path_batch, path_sgd, path_mini))
print("Minimum:", theta_best)
for name, p in [("Batch", path_batch), ("SGD", path_sgd), ("Mini", path_mini)]:
    print(name, "Ende:", p[-1], "Abstand:", np.linalg.norm(p[-1] - theta_best))

fig, ax = plt.subplots(figsize=(11, 5), dpi=100)
ax.plot(path_sgd[:, 0], path_sgd[:, 1], "-s", color="#dc2626", lw=1, ms=4,
        label="SGD (ein Punkt je Schritt)")
ax.plot(path_mini[:, 0], path_mini[:, 1], "-+", color="#15803d", lw=1.8, ms=7,
        label="Mini-Batch (8 Punkte)")
ax.plot(path_batch[:, 0], path_batch[:, 1], "-o", color="#2563eb", lw=3.5, ms=5,
        label="Batch (alle Punkte)")
ax.plot(*theta_best, marker="*", ms=18, color="#fde68a", mec="black", ls="none",
        label="Minimum", zorder=10)
ax.set_xlim(theta_best[0] - 1.0, theta_best[0] + 0.5)
ax.set_ylim(theta_best[1] - 0.8, theta_best[1] + 0.6)
ax.set_xlabel("Achsenabschnitt b", fontsize=13)
ax.set_ylabel("Steigung m", fontsize=13)
ax.grid(alpha=0.4)
ax.legend(loc="lower right", fontsize=13)
fig.tight_layout()
fig.savefig(out)
