import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))   # Figure = Blatt, Axes = Diagramm

ax.hist(df["Age"].dropna(), bins=30)

ax.set_title("Altersverteilung der Passagiere")
ax.set_xlabel("Alter in Jahren")
ax.set_ylabel("Anzahl")

fig.tight_layout()
fig.savefig("alter_histogramm.png", dpi=150)
plt.show()



fig, axes = plt.subplots(
    1, 2, figsize=(11, 6.2))

axes[0].hist(df["Age"].dropna(),
             bins=30)
axes[0].set_title("Alter")
axes[0].set_xlabel("Jahre")
axes[0].set_ylabel("Anzahl")

axes[1].hist(df["Fare"], bins=30,
             color="#ff7f0e")
axes[1].set_title("Ticketpreis")
axes[1].set_xlabel("Fare")

fig.suptitle("Zwei Verteilungen")
fig.tight_layout()



import seaborn as sns

fig, ax = plt.subplots()
sns.histplot(data=df, x="Age",
             bins=30, kde=True,
             ax=ax)
ax.set_title("Altersverteilung")
ax.set_xlabel("Alter in Jahren")
ax.set_ylabel("Anzahl")


spalten = ["Survived", "Pclass", "Age",
           "SibSp", "Parch", "Fare"]
corr = df[spalten].corr()

fig, ax = plt.subplots()
sns.heatmap(corr, annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax)
ax.set_title("Korrelationsmatrix")
