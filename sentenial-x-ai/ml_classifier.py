import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from memory import MEMORY_FILE

MODEL_PATH = "ml_model.pkl"

def train_model():
    # Load past commands
    if not os.path.exists(MEMORY_FILE):
        return
    data, labels = [], []
    import json
    with open(MEMORY_FILE) as f:
        for entry in json.load(f):
            cmd = entry["event"].get("command")
            if cmd:
                data.append(cmd)
                # label by command name
                labels.append(cmd.split(" ", 1)[0])
    if not data: return

    vec = CountVectorizer()
    X = vec.fit_transform(data)
    clf = MultinomialNB()
    clf.fit(X, labels)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((vec, clf), f)

def classify_command(cmd):
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        vec, clf = pickle.load(f)
    label = clf.predict(vec.transform([cmd]))[0]
    return label
