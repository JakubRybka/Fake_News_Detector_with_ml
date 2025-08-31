# Climate Change Fake news detector using NLP Features. 

This project performs **feature-rich** detection of fake news, integrating traditional NLP features, semantic similarity with a factual knowledge base, and machine learning modeling. The goal is to build a system that can classify articles using both syntactic and semantic information.

---

## Project Structure

```
.
├── know_base.csv                  # CSV file with factual statements (knowledge base)
├── Exrtact_WelFake.py             # Extract climate change articles
├── NLP_FE.py                      # NLP feature extraction script
├── Fact_Ratio.py                  # Knowledge base matching script
├── NLP_ML.py                      # Final classification model training script
└── README.md                      # ← You are here
```

---

## Overview

### 1. Climate change articles Extraction (`Extract_WelFake.py`)
- Filtering dataset with use of keywords to find only those related to climate change.
- Saves all articles into `climate_articles.csv`


### 2. NLP Feature Extraction (`NLP_FE.py`)
- Extracts linguistic features from each article using `spaCy`, `TextBlob`, and `NLTK`.
- Includes:
  - POS ratios (nouns, verbs, adjectives)
  - Named entity counts (ORG, GPE, DATE, etc.)
  - Sentiment polarity
  - Lexical complexity (sentence length, word length, syllables)
  - POS n-gram patterns
  - Spelling error ratio
  - TF-IDF representation using a custom vocabulary
- Saves all features into `feature_a.csv`.

### 3. Fact Matching (`Fact_Ratio.py`)
- Loads a knowledge base of factual sentences.
- Encodes each sentence in each article and compares it to the knowledge base using **cosine similarity** via `SentenceTransformers`.
- Calculates a **fact match ratio**: the proportion of article sentences semantically aligned with any known fact.
- Appends this feature to the previous dataset and saves it as `feature_a2.csv`.

### 4. Model Training (`NLP_ML.py`)
- Loads extracted features (`features2.csv` assumed to contain all necessary columns).
- Trains a **Random Forest classifier** (or Logistic Regression optionally).
- Evaluates the model using:
  - **F1 score**
  - **Accuracy**
- `Label` is binary (mapped to 0 or 1 via `1 - x` transformation).

---

##  Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

Also download required NLTK and spaCy assets:

```python
# In Python shell
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# In terminal
python -m spacy download en_core_web_sm
```

You will also need to prepare WelFake dataset for training. Download it from [WelFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification).

---

## Running the Project

### Step 1: Extract NLP Features

```bash
python features1.py
# Outputs: feature_a.csv
```

### Step 2: Add Fact Match Ratio

```bash
python features2.py
# Outputs: feature_a2.csv
```

### Step 3: Train and Evaluate the Model

```bash
python train_model.py
# Outputs F1 score and accuracy
```

---


##  Potential Extensions

- Add  **zero-shot classification** features.
- Introduce **fine-tuned transformer models** for deeper contextual features.
- Perform **explainability** analysis using SHAP or LIME.
- Evaluate model fairness across different subgroups of articles.

---


## Acknowledgements

- [spaCy](https://spacy.io/)
- [NLTK](https://www.nltk.org/)
- [TextBlob](https://textblob.readthedocs.io/en/dev/)
- [Hugging Face Sentence Transformers](https://www.sbert.net/)
