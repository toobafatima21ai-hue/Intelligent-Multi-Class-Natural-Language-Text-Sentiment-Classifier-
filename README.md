# Airline Tweet Sentiment Classifier

A multi-class NLP sentiment classifier that categorizes airline-related tweets as **positive**, **negative**, or **neutral**, built with classical machine learning (Scikit-Learn) and a full preprocessing → feature extraction → model selection → evaluation pipeline.

## Overview

This project classifies unstructured tweet text directed at US airlines into sentiment categories. Rather than picking a single model arbitrarily, it evaluates four candidate classifiers under stratified cross-validation, tunes the best one, and reports evaluation metrics suited to imbalanced multi-class data (macro F1 as the primary metric, alongside weighted/micro F1 and confusion matrices).

## Dataset

**Twitter US Airline Sentiment** (Kaggle / CrowdFlower)
- ~14,600 tweets directed at major US airlines
- Labeled: `positive`, `negative`, `neutral`
- Naturally imbalanced (majority negative), which the pipeline explicitly accounts for
- Source: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

## Pipeline

| Stage | Technique |
|---|---|
| Text cleaning | Lowercasing, URL removal, @mention/#hashtag stripping, number removal, punctuation removal |
| Normalization | Contraction expansion (`don't` → `do not`) |
| Stopword removal | NLTK stopwords, with negation words (`not`, `never`, `no`) preserved |
| Lemmatization | NLTK `WordNetLemmatizer` |
| Feature extraction | TF-IDF (unigrams + bigrams) |
| Model selection | 5-fold Stratified CV across Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest |
| Tuning | `GridSearchCV` on TF-IDF params + best classifier's hyperparameters |
| Evaluation | Accuracy, macro/weighted/micro F1, per-class classification report, confusion matrices (raw + normalized) |

## Project Structure

\```
├── sentiment_classifier.py     # Main pipeline script
├── Tweets.csv                  # Dataset (not included — download from Kaggle)
├── sentiment_pipeline.joblib   # Trained model (generated after running)
├── confusion_matrix.png        # Evaluation visualization (generated after running)
├── requirements.txt
└── README.md
\```

## Installation

\```bash
git clone https://github.com/toobafatima21ai-hue/airline-sentiment-classifier.git
cd airline-sentiment-classifier
pip install -r requirements.txt
\```

**requirements.txt**
\```
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
joblib
\```

## Usage

1. Download `Tweets.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) and place it in the project root.
2. Run the pipeline:

\```bash
python sentiment_classifier.py
\```

This will:
- Preprocess the dataset
- Run cross-validated model comparison
- Tune and select the best model
- Evaluate on a held-out test set
- Save the trained pipeline (`sentiment_pipeline.joblib`) and confusion matrix plot (`confusion_matrix.png`)

### Inference on new text

\```python
import joblib
from sentiment_classifier import preprocess_text

pipeline = joblib.load("sentiment_pipeline.joblib")

text = "The flight was delayed for 3 hours, absolutely frustrating."
clean = preprocess_text(text)
prediction = pipeline.predict([clean])[0]
print(prediction)
\```

 .

## Tech Stack

- **Language:** Python
- **ML:** Scikit-Learn (Logistic Regression, Linear SVM, Naive Bayes, Random Forest)
- **NLP:** NLTK (stopwords, lemmatization, tokenization)
- **Feature Extraction:** TF-IDF
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib

 
## Author

Tooba Fatima — AI Engineering Student
[GitHub](https://github.com/toobafatima21ai-hue)
