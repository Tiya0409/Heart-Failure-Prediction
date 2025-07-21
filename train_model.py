# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Features and target
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("✅ Model trained & saved as model.pkl")
