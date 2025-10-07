## Keywords of Stock Price Movements and Model-Based Prediction


### Project Overview

This project explores how **news sentiment and key terms** impact the stock movement of EVA Air (2618).
We combined **text mining, keyword extraction, and machine learning models** (SVM, RF, Logistic Regression, BERT fine-tuning, PCA, unsupervised clustering, etc.) with **backtesting strategies** to evaluate predictive performance.

---

### Key Findings

* Extracted bullish/bearish keywords using **2–4 gram + χ² + TF-IDF + MI + Lift** combined ranking.
* Logistic Regression baseline with PCA reduced ~90k → 778 dimensions while keeping 95% variance.
* Backtesting (rolling window, 60-day lookback, D+5 horizon) showed ~74% accuracy when trade conditions were triggered.
* BERT fine-tuning further improved contextual understanding of financial news.

---

### Methodology

1. **Data Collection**

   * EVA Air (2618) stock prices from TWSE.
   * News corpus scraped from online sources.

2. **Preprocessing**

   * Tokenization with Jieba, merged title & content.
   * Labeling strategy: weighted D+1/2/3 price movement (>±3%, weights 0.5/0.3/0.2).

3. **Feature Engineering**

   * TF-IDF vectorization, PCA for dimensionality reduction.
   * Exported vectors for downstream models.

4. **Modeling**

   * Classical ML: SVM, Random Forest, KNN, Logistic Regression, Naive Bayes.
   * Neural approaches: BERT fine-tuning, LLM classification, Unsupervised clustering.
   <img width="1214" height="515" alt="截圖 2025-10-02 下午4 31 51" src="https://github.com/user-attachments/assets/3c5a5852-8915-4f4e-9582-dbde6fa66195" />


5. **Backtesting**

   * Rolling training on past 60 days, prediction horizon of 5 days.
   * Trade entry filtered by bullish/bearish news gap threshold.

---

### Business Implications

* Provides a **quantitative framework** for integrating financial news into trading strategies.
* Demonstrates how **AI-driven sentiment analysis** can improve investment decisions.
* Methodology can extend beyond EVA Air to **other listed companies or industries**.

---

### Files Included

* **回測.py / 回測_出手條件.py** → Backtesting framework with trade-entry rules.
* **Label.py** → Labeling news articles based on stock movements.
* **export_vector.py** → Export TF-IDF / embedding vectors.
* **PCA.py / Option3_PCA.py** → PCA dimensionality reduction experiments.
* **Option3.py** → Alternative modeling pipeline.
* **Unsupervised.py** → Clustering analysis.
* **BERT.py / BERT_finetune.py** → BERT embedding & fine-tuning for text classification.
* **LLM.py** → Large language model experiments.
* **related_articles.py** → Fetching or linking related news.

