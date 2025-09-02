import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

# Dummy dataset for baseline model
X = [
    "DROP TABLE users;",
    "<script>alert('XSS')</script>",
    "SELECT * FROM accounts WHERE id = 1",
    "wget http://malware.com/file",
    "normal login request",
    "hello world",
    "get user profile",
]
y = [1, 1, 1, 1, 0, 0, 0]  # 1 = malicious, 0 = benign

vectorizer = HashingVectorizer(n_features=512, alternate_sign=False, norm="l2", lowercase=True)
X_vec = vectorizer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Model accuracy:", model.score(X_test, y_test))

# Save model
joblib.dump(model, "services/threat-engine/models/threat_model.pkl")
