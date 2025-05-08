# Fake-Article-Detection-System

DESCRIPTION:
This Python-based system detects whether a given article is fake or real using natural language processing (NLP) and machine learning techniques.

The model uses a TF-IDF vectorizer and a pre-trained classifier (e.g., Logistic Regression or PassiveAggressiveClassifier) to classify articles.

REQUIREMENTS:
- Download fake_news_model.pkl and tfidf_vectorizer.pkl from Kaggle platform

FEATURES:
- Input can be a single article or batch processing from text files.
- Outputs "FAKE" or "REAL" label.
- Can be extended to support PDFs, Word docs, or website URLs.

USAGE:
1. Train the model (optional, if you're not using pre-trained files).
2. Run the detection script:

   ```bash
   python detect_fake_article.py
