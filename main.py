import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


df = pd.read_csv("student_dataset_v3.csv", sep=";")

# عرض أول 5 صفوف
print(df.head())

df["reussite"] = df["reussite"].map({"yes": 1, "no": 0})

# حذف الصفوف المكررة
df = df.drop_duplicates()




if "note de smestre1" in df.columns:
    df.rename(columns={"note de smestre1": "g1"}, inplace=True)
if "heures-etudie" in df.columns:
    df.rename(columns={"heures-etudie": "study_hours"}, inplace=True)

# Keep the model aligned with the current app inputs.
features = ["age", "absences", "g1", "study_hours"]
X = df[features]
y = df["reussite"]

# For success probability, enforce the expected directions in tree models:
# age: no rule, absences: lower is better, g1/study_hours: higher is better.
monotonic_constraints = [0, -1, 1, 1]


imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# تأكد أنه لا يوجد NaN
print("Valeurs manquantes après imputation:\n", X.isna().sum())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
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

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n==============================")
    print(name)
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    trained_models[name] = model

joblib.dump(trained_models, 'student_models.pkl')
joblib.dump(imputer, 'imputer.pkl')

print(f"\nSaved {len(trained_models)} models to student_models.pkl")


df_imp = pd.DataFrame({
    "feature": X.columns,
    "coefficient": trained_models["Logistic Regression"].coef_[0]
}).sort_values(by="coefficient", ascending=False)

print("\nFeature Effects (Logistic Regression):\n", df_imp)

# Feature importances for tree-based models
for name in ["Decision Tree", "Random Forest"]:
    if hasattr(trained_models[name], 'feature_importances_'):
        df_imp_tree = pd.DataFrame({
            "feature": X.columns,
            "importance": trained_models[name].feature_importances_
        }).sort_values(by="importance", ascending=False)
        print(f"\nFeature Importances ({name}):\n", df_imp_tree)



# Presentation graphics des donnes ;
#sns.countplot(x="reussite", data=df)
#plt.show()

# العلاقة بين الغيابات والنجاح
#sns.boxplot(x="reussite", y="absences", data=df)
#plt.show()

# خريطة الارتباط
#sns.heatmap(df.corr(), annot=True)
#plt.show()
