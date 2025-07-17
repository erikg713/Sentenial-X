def train_model(texts, labels, random_state=42):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(random_state=random_state)
    model.fit(X, labels)
    return model, vectorizer

