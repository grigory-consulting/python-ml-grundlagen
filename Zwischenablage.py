import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))   # Figure = Blatt, Axes = Diagramm

ax.hist(df["Age"].dropna(), bins=30)

ax.set_title("Altersverteilung der Passagiere")
ax.set_xlabel("Alter in Jahren")
ax.set_ylabel("Anzahl")

fig.tight_layout()
fig.savefig("alter_histogramm.png", dpi=150)
plt.show()
