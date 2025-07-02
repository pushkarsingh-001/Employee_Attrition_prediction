# -----------------------------------------
# IMPORT LIBRARIES
# -----------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
# replace path with your CSV or Excel file
df = pd.read_csv("Employee_Attrition.csv")

print("Data Shape:", df.shape)
print(df.head())

# -----------------------------------------
# BASIC EDA
# -----------------------------------------
print(df.info())
print(df.describe())

# Check nulls
print(df.isnull().sum())

# Attrition value counts
print(df['Attrition'].value_counts())

# Plot target distribution
sns.countplot(x='Attrition', data=df)
plt.title("Attrition Distribution")
plt.show()

# -----------------------------------------
# CORRELATION MATRIX
# -----------------------------------------
plt.figure(figsize=(12,10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------------------
# ENCODING CATEGORICAL VARIABLES
# -----------------------------------------

# Identify categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical Columns:", cat_cols)

# Encode categorical features
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print(df.head())

# -----------------------------------------
# TRAIN-TEST SPLIT
# -----------------------------------------
X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------
# FEATURE SCALING
# -----------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------
# MODEL TRAINING
# -----------------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# -----------------------------------------
# PREDICTIONS
# -----------------------------------------
y_pred = rf.predict(X_test_scaled)

# -----------------------------------------
# MODEL EVALUATION
# -----------------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# -----------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.sort_values().plot(kind='barh', figsize=(8,6))
plt.title("Feature Importances")
plt.show()