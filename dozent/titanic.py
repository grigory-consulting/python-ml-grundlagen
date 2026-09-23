"""Titanic-Klassifikation, Zielgröße Survived, Metrik Accuracy.

Jedes Experiment wird einmal ausgewertet (5-fach Stratified-CV auf dem
Trainingsteil plus Holdout-Test) und als Zeile an results.csv angehängt.
Bereits protokollierte Experimente werden beim erneuten Lauf übersprungen.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)
from sklearn.svm import SVC

RESULTS_PATH = "results.csv"
RANDOM_STATE = 42

df = pd.read_csv("data/titanic.csv")
y = df["Survived"]
X = df.drop(columns=["Survived"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
# Für die Endrunde: 5x5 wiederholte CV, glättet den Zufall der Fold-Einteilung
repeated_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)


NUM_COLS = ["Age", "SibSp", "Parch", "Fare"]
CAT_COLS = ["Pclass", "Sex", "Embarked"]


def basic_preprocessor():
    """Numerik: Median-Imputation + Skalierung. Kategorien: häufigster Wert + One-Hot.
    Name, Ticket, Cabin, PassengerId fallen weg (remainder="drop")."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        [("num", numeric, NUM_COLS), ("cat", categorical, CAT_COLS)],
        remainder="drop",
    )


def add_features(X):
    """Abgeleitete Merkmale: Anrede aus dem Namen, Familiengröße, Allein-Reisend, Kabine bekannt."""
    X = X.copy()
    title = X["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    common = {"Mr", "Mrs", "Miss", "Master"}
    X["Title"] = title.where(title.isin(common), "Rare")
    X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
    X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
    X["HasCabin"] = X["Cabin"].notna().astype(int)
    return X


FE_NUM_COLS = NUM_COLS + ["FamilySize", "IsAlone", "HasCabin"]
FE_CAT_COLS = CAT_COLS + ["Title"]


def fe_preprocessor():
    """Wie basic_preprocessor, aber mit den abgeleiteten Merkmalen aus add_features."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    columns = ColumnTransformer(
        [("num", numeric, FE_NUM_COLS), ("cat", categorical, FE_CAT_COLS)],
        remainder="drop",
    )
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("columns", columns),
    ])


class GroupFeatures(BaseEstimator, TransformerMixin):
    """Merkmale, die auf dem Trainingsteil gelernt werden müssen:
    Ticket-Häufigkeit, Fare pro Person, Deck aus Cabin, Alter nach Anrede imputiert,
    Nachname und Ticket als Kategorien (für TargetEncoder)."""

    def fit(self, X, y=None):
        X = add_features(X)
        self.ticket_counts_ = X["Ticket"].value_counts()
        self.age_by_title_ = X.groupby("Title")["Age"].median()
        self.age_global_ = X["Age"].median()
        return self

    def transform(self, X):
        X = add_features(X)
        X["TicketFreq"] = X["Ticket"].map(self.ticket_counts_).fillna(1).astype(int)
        X["FarePerPerson"] = X["Fare"] / X["TicketFreq"]
        X["LogFare"] = np.log1p(X["Fare"].fillna(0))
        X["Deck"] = X["Cabin"].str[0].fillna("U")
        age_fill = X["Title"].map(self.age_by_title_).fillna(self.age_global_)
        X["AgeMissing"] = X["Age"].isna().astype(int)
        X["Age"] = X["Age"].fillna(age_fill)
        X["IsChild"] = (X["Age"] < 14).astype(int)
        X["Surname"] = X["Name"].str.split(",").str[0].str.strip()
        return X


FE2_NUM_COLS = FE_NUM_COLS + ["TicketFreq", "FarePerPerson", "LogFare", "AgeMissing", "IsChild"]
FE2_CAT_COLS = FE_CAT_COLS + ["Deck"]
FE2_TARGET_COLS = ["Surname", "Ticket"]


def fe2_preprocessor(target_encode=False):
    """Stufe 2: GroupFeatures plus optional TargetEncoder auf Nachname und Ticket.
    TargetEncoder nutzt intern Cross-Fitting, daher kein Zielleck innerhalb der Pipeline."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
    ])
    transformers = [("num", numeric, FE2_NUM_COLS), ("cat", categorical, FE2_CAT_COLS)]
    if target_encode:
        transformers.append((
            "target",
            TargetEncoder(target_type="binary", smooth="auto", random_state=RANDOM_STATE),
            FE2_TARGET_COLS,
        ))
    columns = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("features", GroupFeatures()), ("columns", columns)])


SLIM_NUM_COLS = ["Age", "Fare", "FamilySize"]
SLIM_CAT_COLS = ["Pclass", "Sex", "Title"]


def slim_preprocessor():
    """Bewusst wenige Merkmale: Pclass, Sex, Title, Age, Fare, FamilySize."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore")
    columns = ColumnTransformer(
        [("num", numeric, SLIM_NUM_COLS), ("cat", categorical, SLIM_CAT_COLS)],
        remainder="drop",
    )
    return Pipeline([("features", FunctionTransformer(add_features)), ("columns", columns)])


def top_candidates():
    """Die aussichtsreichsten Modelle für die Endrunde mit wiederholter CV."""
    return {
        "voting_soft_fe": Pipeline([
            ("prep", fe_preprocessor()),
            ("clf", VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000)),
                    ("rf", RandomForestClassifier(
                        n_estimators=300, max_depth=6, min_samples_leaf=3,
                        random_state=RANDOM_STATE)),
                    ("hgb", HistGradientBoostingClassifier(
                        learning_rate=0.03, max_depth=3, random_state=RANDOM_STATE)),
                    ("svc", SVC(probability=True, random_state=RANDOM_STATE)),
                ],
                voting="soft",
            )),
        ]),
        "svc_rbf_fe": Pipeline([
            ("prep", fe_preprocessor()),
            ("clf", SVC(C=1.0, gamma="scale", random_state=RANDOM_STATE)),
        ]),
        "logreg_fe": Pipeline([
            ("prep", fe_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "hgb_fe2_te": Pipeline([
            ("prep", fe2_preprocessor(target_encode=True)),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.03, max_depth=3, random_state=RANDOM_STATE)),
        ]),
        "hgb_fe": Pipeline([
            ("prep", fe_preprocessor()),
            ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }


def build_experiments():
    """Liste (name, estimator[, cv]). Neue Versuche hier unten anhängen."""
    return [
        ("dummy_most_frequent", DummyClassifier(strategy="most_frequent")),
        ("logreg_basic", Pipeline([
            ("prep", basic_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000)),
        ])),
        ("rf_basic", Pipeline([
            ("prep", basic_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ])),
        ("hgb_basic", Pipeline([
            ("prep", basic_preprocessor()),
            ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ])),
        ("logreg_fe", Pipeline([
            ("prep", fe_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000)),
        ])),
        ("rf_fe", Pipeline([
            ("prep", fe_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ])),
        ("hgb_fe", Pipeline([
            ("prep", fe_preprocessor()),
            ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ])),
        ("svc_rbf_fe", Pipeline([
            ("prep", fe_preprocessor()),
            ("clf", SVC(C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE)),
        ])),
        # GridSearchCV als Estimator: cross_val_score außen, Tuning innen (geschachtelte CV).
        ("rf_fe_tuned", GridSearchCV(
            Pipeline([
                ("prep", fe_preprocessor()),
                ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
            ]),
            param_grid={
                "clf__max_depth": [4, 6, 8, None],
                "clf__min_samples_leaf": [1, 3, 5],
            },
            cv=inner_cv, scoring="accuracy", n_jobs=-1,
        )),
        ("hgb_fe_tuned", GridSearchCV(
            Pipeline([
                ("prep", fe_preprocessor()),
                ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]),
            param_grid={
                "clf__learning_rate": [0.03, 0.1],
                "clf__max_depth": [3, 5, None],
                "clf__max_iter": [100, 300],
            },
            cv=inner_cv, scoring="accuracy", n_jobs=-1,
        )),
        ("logreg_fe_tuned", GridSearchCV(
            Pipeline([
                ("prep", fe_preprocessor()),
                ("clf", LogisticRegression(max_iter=2000)),
            ]),
            param_grid={"clf__C": [0.03, 0.1, 0.3, 1, 3, 10]},
            cv=inner_cv, scoring="accuracy", n_jobs=-1,
        )),
        ("svc_rbf_fe_tuned", GridSearchCV(
            Pipeline([
                ("prep", fe_preprocessor()),
                ("clf", SVC(random_state=RANDOM_STATE)),
            ]),
            param_grid={
                "clf__C": [0.3, 1, 3, 10],
                "clf__gamma": ["scale", 0.03, 0.1],
            },
            cv=inner_cv, scoring="accuracy", n_jobs=-1,
        )),
        ("voting_soft_fe", Pipeline([
            ("prep", fe_preprocessor()),
            ("clf", VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000)),
                    ("rf", RandomForestClassifier(
                        n_estimators=300, max_depth=6, min_samples_leaf=3,
                        random_state=RANDOM_STATE)),
                    ("hgb", HistGradientBoostingClassifier(
                        learning_rate=0.03, max_depth=3, random_state=RANDOM_STATE)),
                    ("svc", SVC(probability=True, random_state=RANDOM_STATE)),
                ],
                voting="soft",
            )),
        ])),
        # Stufe 2: Gruppenmerkmale
        ("logreg_fe2", Pipeline([
            ("prep", fe2_preprocessor()),
            ("clf", LogisticRegression(C=0.3, max_iter=2000)),
        ])),
        ("hgb_fe2", Pipeline([
            ("prep", fe2_preprocessor()),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.03, max_depth=3, random_state=RANDOM_STATE)),
        ])),
        ("svc_rbf_fe2", Pipeline([
            ("prep", fe2_preprocessor()),
            ("clf", SVC(C=1.0, gamma="scale", random_state=RANDOM_STATE)),
        ])),
        ("extratrees_fe2", Pipeline([
            ("prep", fe2_preprocessor()),
            ("clf", ExtraTreesClassifier(
                n_estimators=500, max_depth=8, min_samples_leaf=2, random_state=RANDOM_STATE)),
        ])),
        ("knn_fe2", Pipeline([
            ("prep", fe2_preprocessor()),
            ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance")),
        ])),
        ("mlp_fe2", Pipeline([
            ("prep", fe2_preprocessor()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=2000,
                early_stopping=True, random_state=RANDOM_STATE)),
        ])),
        # Stufe 2 plus Zielkodierung von Nachname und Ticket
        ("logreg_fe2_te", Pipeline([
            ("prep", fe2_preprocessor(target_encode=True)),
            ("clf", LogisticRegression(C=0.3, max_iter=2000)),
        ])),
        ("hgb_fe2_te", Pipeline([
            ("prep", fe2_preprocessor(target_encode=True)),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.03, max_depth=3, random_state=RANDOM_STATE)),
        ])),
        ("svc_rbf_fe2_te", Pipeline([
            ("prep", fe2_preprocessor(target_encode=True)),
            ("clf", SVC(C=1.0, gamma="scale", random_state=RANDOM_STATE)),
        ])),
        ("stacking_fe2", Pipeline([
            ("prep", fe2_preprocessor()),
            ("clf", StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression(C=0.3, max_iter=2000)),
                    ("rf", RandomForestClassifier(
                        n_estimators=300, max_depth=6, min_samples_leaf=3,
                        random_state=RANDOM_STATE)),
                    ("hgb", HistGradientBoostingClassifier(
                        learning_rate=0.03, max_depth=3, random_state=RANDOM_STATE)),
                    ("svc", SVC(probability=True, random_state=RANDOM_STATE)),
                ],
                final_estimator=LogisticRegression(max_iter=1000),
                cv=inner_cv, stack_method="predict_proba",
            )),
        ])),
        # Schlanke Merkmalsmenge
        ("logreg_slim", Pipeline([
            ("prep", slim_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000)),
        ])),
        ("hgb_slim", Pipeline([
            ("prep", slim_preprocessor()),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.03, max_depth=3, random_state=RANDOM_STATE)),
        ])),
        ("svc_rbf_slim", Pipeline([
            ("prep", slim_preprocessor()),
            ("clf", SVC(C=1.0, gamma="scale", random_state=RANDOM_STATE)),
        ])),
        # Endrunde: dieselben Modelle, aber 5x5 wiederholte CV
        *[(f"{name}_rep5x5", est, repeated_cv) for name, est in top_candidates().items()],
    ]


def already_logged():
    """Namen der Experimente, die schon in results.csv stehen."""
    if not os.path.exists(RESULTS_PATH):
        return set()
    return set(pd.read_csv(RESULTS_PATH)["experiment"])


def evaluate(name, estimator, cv_obj=None):
    """CV-Accuracy auf dem Trainingsteil, dann Fit auf ganzem Trainingsteil und Holdout."""
    scores = cross_val_score(
        estimator, X_train, y_train, cv=cv_obj or cv, scoring="accuracy", n_jobs=-1
    )
    estimator.fit(X_train, y_train)
    test_acc = estimator.score(X_test, y_test)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": name,
        "cv_accuracy_mean": round(scores.mean(), 4),
        "cv_accuracy_std": round(scores.std(), 4),
        "test_accuracy": round(test_acc, 4),
    }


def main():
    done = already_logged()
    rows = []
    for name, estimator, *extra in build_experiments():
        if name in done:
            continue
        row = evaluate(name, estimator, *extra)
        rows.append(row)
        print(f"{name:32s} cv={row['cv_accuracy_mean']:.4f}±{row['cv_accuracy_std']:.4f} "
              f"test={row['test_accuracy']:.4f}")
    if rows:
        pd.DataFrame(rows).to_csv(
            RESULTS_PATH, mode="a", index=False, header=not os.path.exists(RESULTS_PATH)
        )
    print("\nStand results.csv:")
    print(pd.read_csv(RESULTS_PATH).to_string(index=False))


if __name__ == "__main__":
    main()
