"""
  Intelligent Multi-Class NLP Sentiment Classifier
Dataset: Twitter US Airline Sentiment (Kaggle)
-------------------------------------------------------------------------
Pipeline: Clean -> Expand contractions -> Tokenize -> Stopword removal
          (negation-aware) -> Lemmatize -> TF-IDF -> Model selection via
          CV -> Hyperparameter tuning -> Evaluation -> Persistence

Dataset columns used: "text" (tweet), "airline_sentiment" (positive/
negative/neutral). Download from:
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
File name: Tweets.csv
"""

import re
import string
import warnings
import numpy as np
import pandas as pd
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 0. NLTK setup
# ------------------------------------------------------------------
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(
            f"corpora/{resource}" if resource in ("stopwords", "wordnet", "omw-1.4")
            else f"tokenizers/{resource}"
        )
    except LookupError:
        nltk.download(resource, quiet=True)

LEMMATIZER = WordNetLemmatizer()

BASE_STOPWORDS = set(stopwords.words("english"))
NEGATIONS = {"not", "no", "nor", "never", "n't", "cannot", "none", "nothing"}
STOP_WORDS = BASE_STOPWORDS - NEGATIONS

CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "wouldn't": "would not", "can't": "cannot",
    "couldn't": "could not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "haven't": "have not",
    "hasn't": "has not", "hadn't": "had not", "shouldn't": "should not",
    "i'm": "i am", "it's": "it is", "that's": "that is",
    "there's": "there is", "i've": "i have", "i'll": "i will",
}


# ------------------------------------------------------------------
# 1. Load Twitter US Airline Sentiment dataset
# ------------------------------------------------------------------
DATA_PATH = "Tweets.csv"  # update path if needed

df_raw = pd.read_csv(DATA_PATH)
df = df_raw[["text", "airline_sentiment"]].rename(
    columns={"airline_sentiment": "label"}
)
df = df.dropna(subset=["text", "label"])
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("Class distribution (raw):")
print(df["label"].value_counts())
print(f"\nTotal samples: {len(df)}")


# ------------------------------------------------------------------
# 2. Preprocessing (Twitter-specific additions: @mentions, hashtags)
# ------------------------------------------------------------------
def expand_contractions(text: str) -> str:
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = expand_contractions(text)
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"@\w+", "", text)                     # @mentions (e.g. @united)
    text = re.sub(r"#", "", text)                         # keep hashtag word, drop symbol
    text = re.sub(r"\d+", "", text)                       # numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 1
    ]
    return " ".join(tokens)


print("\nPreprocessing tweets...")
df["clean_text"] = df["text"].apply(preprocess_text)
df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

print(df[["text", "clean_text"]].head(3).to_string(index=False))


# ------------------------------------------------------------------
# 3. Train/test split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"],
)


# ------------------------------------------------------------------
# 4. Model selection via Stratified K-Fold CV
#    Note: this dataset is imbalanced (~63% negative, ~21% neutral,
#    ~16% positive), so class_weight="balanced" is used throughout
#    and macro F1 (not accuracy) is the primary selection metric.
# ------------------------------------------------------------------
candidates = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "LinearSVC": LinearSVC(class_weight="balanced"),
    "MultinomialNB": MultinomialNB(),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
vectorizer_probe = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
X_train_probe = vectorizer_probe.fit_transform(X_train)

print("\n" + "=" * 60)
print("MODEL SELECTION — 5-Fold CV Macro F1 (mean ± std)")
print("=" * 60)
cv_results = {}
for name, clf in candidates.items():
    scores = cross_val_score(clf, X_train_probe, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:20s}: {scores.mean():.4f} ± {scores.std():.4f}")

best_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\nBest model by CV macro F1: {best_model_name}")


# ------------------------------------------------------------------
# 5. Hyperparameter tuning (TF-IDF + best classifier, jointly)
# ------------------------------------------------------------------
param_grids = {
    "LogisticRegression": {
        "clf": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "params": {
            "clf__C": [0.1, 1, 10],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__max_features": [5000, 8000],
        },
    },
    "LinearSVC": {
        "clf": LinearSVC(class_weight="balanced"),
        "params": {
            "clf__C": [0.1, 1, 10],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__max_features": [5000, 8000],
        },
    },
    "MultinomialNB": {
        "clf": MultinomialNB(),
        "params": {
            "clf__alpha": [0.1, 0.5, 1.0],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__max_features": [5000, 8000],
        },
    },
    "RandomForest": {
        "clf": RandomForestClassifier(class_weight="balanced", random_state=42),
        "params": {
            "clf__n_estimators": [200, 300],
            "clf__max_depth": [None, 30],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
        },
    },
}

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=2)),
    ("clf", param_grids[best_model_name]["clf"]),
])

grid = GridSearchCV(
    pipeline,
    param_grid=param_grids[best_model_name]["params"],
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train, y_train)

print(f"\nBest params: {grid.best_params_}")
print(f"Best CV macro F1: {grid.best_score_:.4f}")

best_pipeline = grid.best_estimator_


# ------------------------------------------------------------------
# 6. Final evaluation on held-out test set
# ------------------------------------------------------------------
y_pred = best_pipeline.predict(X_test)

print("\n" + "=" * 60)
print("FINAL TEST SET EVALUATION")
print("=" * 60)
print(f"Accuracy:          {accuracy_score(y_test, y_pred):.4f}")
print(f"Macro F1-score:    {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Weighted F1-score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Micro F1-score:    {f1_score(y_test, y_pred, average='micro'):.4f}")

print("\nPer-class report:")
print(classification_report(y_test, y_pred, zero_division=0))

labels_order = sorted(df["label"].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels_order)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_order, yticklabels=labels_order, ax=axes[0])
axes[0].set_title("Confusion Matrix (counts)")
axes[0].set_ylabel("True")
axes[0].set_xlabel("Predicted")

sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels_order, yticklabels=labels_order, ax=axes[1])
axes[1].set_title("Confusion Matrix (normalized)")
axes[1].set_ylabel("True")
axes[1].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nConfusion matrices saved as confusion_matrix.png")


# ------------------------------------------------------------------
# 7. Persist the trained pipeline
# ------------------------------------------------------------------
joblib.dump(best_pipeline, "sentiment_pipeline.joblib")
print("Model saved as sentiment_pipeline.joblib")


# ------------------------------------------------------------------
# 8. Inference function
# ------------------------------------------------------------------
def predict_sentiment(text: str, pipeline=best_pipeline) -> dict:
    clean = preprocess_text(text)
    pred = pipeline.predict([clean])[0]
    result = {"text": text, "predicted_label": pred}
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        proba = pipeline.predict_proba([clean])[0]
        result["confidence"] = dict(zip(pipeline.classes_, np.round(proba, 3)))
    return result


sample_inputs = [
    "@united thank you for the amazing service today!",
    "@AmericanAir my flight got delayed 5 hours, unacceptable.",
    "Flight scheduled to depart at 6pm from gate 22.",
    "@JetBlue not happy with how this was handled.",
]

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)
for text in sample_inputs:
    print(predict_sentiment(text))