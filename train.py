import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("dataset.csv")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(data["label"])

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text → embeddings
X = model.encode(data["text"].tolist())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Print accuracy
print("Train accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))

# Save model + encoder
joblib.dump(clf, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\n✅ Model saved as model.pkl")

data["label"] = data["label"].str.strip().str.lower()