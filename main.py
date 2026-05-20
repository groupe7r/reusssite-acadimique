import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt
import joblib


# =========================
# Lecture du dataset
# =========================

df = pd.read_csv("student_dataset_v3.csv", sep=";")

print(df.head())


# =========================
# Conversion yes/no -> 1/0
# =========================

df["reussite"] = df["reussite"].map({
    "yes": 1,
    "no": 0
})


# =========================
# Suppression des doublons
# =========================

df = df.drop_duplicates()


# =========================
# Renommage des colonnes
# =========================

if "note de smestre1" in df.columns:
    df.rename(columns={
        "note de smestre1": "g1"
    }, inplace=True)

if "heures-etudie" in df.columns:
    df.rename(columns={
        "heures-etudie": "study_hours"
    }, inplace=True)


# =========================
# Features et target
# =========================

features = [
    "age",
    "absences",
    "g1",
    "study_hours"
]

X = df[features]
y = df["reussite"]


# =========================
# Contraintes monotoniques
# =========================

monotonic_constraints = [0, -1, 1, 1]


# =========================
# Gestion des valeurs manquantes
# =========================

imputer = SimpleImputer(strategy='mean')

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

print("\nValeurs manquantes après imputation :")
print(X.isna().sum())


# =========================
# Split Train / Test
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# Modèles
# =========================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight='balanced',
        random_state=42,
        max_depth=5,
        min_samples_leaf=3,
        monotonic_cst=monotonic_constraints
    ),

    "Random Forest": RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        monotonic_cst=monotonic_constraints
    )
}


trained_models = {}


# =========================
# Cross Validation Setup
# =========================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =========================
# Training + Evaluation
# =========================

for name, model in models.items():

    print("\n==============================")
    print(name)

    # ===== Cross Validation =====

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring='accuracy'
    )

    print("\nCross Validation Scores:")
    print(cv_scores)

    print("Mean CV Accuracy:",
          cv_scores.mean())

    print("Std CV:",
          cv_scores.std())

    # ===== Training =====

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\nTest Accuracy:", acc)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    trained_models[name] = model


# =========================
# Sauvegarde des modèles
# =========================

joblib.dump(
    trained_models,
    'student_models.pkl'
)

joblib.dump(
    imputer,
    'imputer.pkl'
)

print(f"\nSaved {len(trained_models)} models to student_models.pkl")


# =========================
# Importance des variables
# =========================

df_imp = pd.DataFrame({

    "feature": X.columns,

    "coefficient":
    trained_models[
        "Logistic Regression"
    ].coef_[0]

}).sort_values(
    by="coefficient",
    ascending=False
)

print("\nFeature Effects (Logistic Regression):")
print(df_imp)


# =========================
# Feature Importances Trees
# =========================

for name in [
    "Decision Tree",
    "Random Forest"
]:

    if hasattr(
        trained_models[name],
        'feature_importances_'
    ):

        df_imp_tree = pd.DataFrame({

            "feature": X.columns,

            "importance":
            trained_models[name].feature_importances_

        }).sort_values(
            by="importance",
            ascending=False
        )

        print(f"\nFeature Importances ({name}):")
        print(df_imp_tree)


# =========================
# Visualisations
# =========================

# Distribution réussite
# sns.countplot(x="reussite", data=df)
# plt.show()

# Relation absences / réussite
# sns.boxplot(x="reussite", y="absences", data=df)
# plt.show()

# Heatmap corrélation
# sns.heatmap(df.corr(), annot=True)
# plt.show()