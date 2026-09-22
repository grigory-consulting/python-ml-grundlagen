"""k-nächste Nachbarn auf Iris: Trainingsblüten, eine neue Blüte und ihre fünf nächsten Nachbarn."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ohne Argument landet das Bild neben dem Skript
out = sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).with_suffix(".png")

# Split wie auf der Folie, aber nur zwei Merkmale, damit man es zeichnen kann
iris = load_iris()
X, y = iris.data[:, 2:], iris.target            # Blütenblatt: Länge, Breite
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
k = 5
model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

neu = np.array([[4.9, 1.6]])                    # eine neue Blüte, nahe der Grenze
abstand, idx = model.kneighbors(neu)
idx = idx[0]
vorhersage = model.predict(neu)[0]
stimmen = np.bincount(y_train[idx], minlength=3)

FARBEN = ["#2459a6", "#15803d", "#dc2626"]
NAMEN = ["setosa", "versicolor", "virginica"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=100, gridspec_kw={"width_ratios": [1.15, 1]})

# links: Trainingsdaten, neue Blüte, Nachbarn
ax = axes[0]
for c in (1, 2):                                # Ausschnitt: setosa liegt weit links außerhalb
    m = y_train == c
    ax.scatter(X_train[m, 0], X_train[m, 1], s=60, color=FARBEN[c], alpha=0.6,
               edgecolors="white", linewidths=0.5, label=NAMEN[c])
for i in idx:
    ax.plot([neu[0, 0], X_train[i, 0]], [neu[0, 1], X_train[i, 1]], color="#334155", lw=1.2, zorder=3)
    ax.scatter(X_train[i, 0], X_train[i, 1], s=200, facecolors="none", edgecolors="#0f172a", linewidths=1.6, zorder=4)
ax.scatter(*neu[0], s=220, marker="*", color="#fde68a", edgecolors="#0f172a", linewidths=1.2, zorder=5, label="neue Blüte")
ax.set_xlabel("Blütenblattlänge in cm", fontsize=12)
ax.set_ylabel("Blütenblattbreite in cm", fontsize=12)
ax.set_title(f"Die {k} nächsten Nachbarn der neuen Blüte (Ausschnitt)", fontsize=13)
ax.set_xlim(3.8, 6.0); ax.set_ylim(1.0, 2.3)
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3)

# rechts: Entscheidungsgebiete desselben Modells
ax = axes[1]
xx, yy = np.meshgrid(np.linspace(0.5, 7.2, 300), np.linspace(-0.2, 2.8, 300))
zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5], colors=FARBEN, alpha=0.18)
for c in range(3):
    m = y_train == c
    ax.scatter(X_train[m, 0], X_train[m, 1], s=22, color=FARBEN[c], alpha=0.7, edgecolors="white", linewidths=0.4)
ax.scatter(*neu[0], s=220, marker="*", color="#fde68a", edgecolors="#0f172a", linewidths=1.2, zorder=5)
ax.set_xlabel("Blütenblattlänge in cm", fontsize=12)
ax.set_title(f"Was das Modell überall vorhersagen würde (k = {k})", fontsize=13)
ax.set_xlim(0.5, 7.2); ax.set_ylim(-0.2, 2.8)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(out)
print("Nachbarn:", [NAMEN[c] for c in y_train[idx]], "| Stimmen:", dict(zip(NAMEN, stimmen)), "| Vorhersage:", NAMEN[vorhersage])
print("Abstände:", abstand[0].round(2))
