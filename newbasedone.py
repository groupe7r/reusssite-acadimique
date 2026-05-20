import pandas as pd

# =========================
# 1. CHARGER LA BASE (séparateur ;)
# =========================
df = pd.read_csv("student_performance_cleaned.csv", sep=",")


print(df.columns)
print("\nAperçu initial :")
print(df.head())

print("\nInfos :")
print(df.info())

# =========================
# 2. SUPPRIMER DOUBLONS
# =========================
df = df.drop_duplicates()

# =========================
# 3. GÉRER LES VALEURS MANQUANTES
# =========================
df = df.dropna()

# =========================
# 4. CORRIGER TYPES
# =========================
df["age"] = df["age"].astype(int)
df["heures-etudie"] = df["heures-etudie"].astype(int)
df["absences"] = df["absences"].astype(int)
df["note de smestre1"] = df["note de smestre1"].astype(int)


# =========================
# 5. NETTOYAGE OUTLIERS
# =========================
df = df[(df["absences"] >= 0) & (df["absences"] <= 30)]
df = df[(df["note de smestre1"] >= 0) & (df["note de smestre1"] <= 20)]

# =========================
# 6. TRANSFORMATION TARGET
# =========================
df["reussite"] = df["reussite"].map({1: "yes", 0: "no"})

# =========================
# 7. FEATURE ENGINEERING
# =========================
df["score_total"] = df["note de smestre1"] - (df["absences"] * 0.2)

# =========================
# 8. SAUVEGARDER NOUVELLE BASE
# =========================
df.to_csv("student_dataset_transformed.csv", sep=";", index=False)

print("\n✔ Base transformée sauvegardée avec séparateur ;")