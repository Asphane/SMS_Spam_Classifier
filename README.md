# 📩 SMS Spam Classification using NLP & Machine Learning

An **end-to-end Natural Language Processing (NLP) project** that classifies SMS messages as **Spam** or **Ham** using classical text vectorization techniques (**Bag of Words & TF-IDF**) combined with multiple **machine learning algorithms**.  
This project focuses on **clean preprocessing, correct ML pipelines, and comparative evaluation**, making it **placement- and interview-ready**.

---

## 🚀 Project Overview

Spam messages are a common real-world problem. This project builds a **robust SMS spam detection system** by:

- Cleaning raw SMS text
- Converting text into numerical features
- Training multiple ML models
- Comparing their performance using standard metrics

---

## 📂 Dataset

**SMS Spam Collection Dataset**

- Each message is labeled as:
  - `ham` → Normal message
  - `spam` → Promotional / fraudulent message
- Real-world, noisy SMS data

**Format**

---

## 🧠 Models & Feature Extraction

### 🔹 Text Vectorization Techniques
- **Bag of Words (BoW)**
- **TF-IDF**

### 🔹 Machine Learning Models
- Naive Bayes
- Logistic Regression
- Support Vector Machine (Linear SVM)

---

## 🛠️ Tech Stack

- **Language**: Python 🐍
- **Libraries**:
  - pandas, numpy
  - nltk
  - scikit-learn
  - regex

---

## 🔄 NLP Pipeline

```

Raw SMS Text
↓
Text Cleaning (regex, lowercase, stopwords, stemming)
↓
Train–Test Split (before vectorization ✅)
↓
Feature Extraction (BoW / TF-IDF)
↓
Model Training (NB / LogReg / SVM)
↓
Evaluation & Comparison

```

---

## 🧹 Text Preprocessing Details

✔ Lowercasing text  
✔ Removing numbers, emojis, and symbols using regex  
✔ Tokenization  
✔ Stopword removal  
✔ Stemming using Porter Stemmer  
✔ Removal of empty texts to prevent TF-IDF issues  

---

## 📊 Model Performance Summary

All models were evaluated using **Accuracy, Precision, Recall, and F1-Score**.  
Below is the **final consolidated result table**:

| Model | Accuracy | Precision | Recall | F1-Score |
|-----|---------|-----------|--------|---------|
| Naive Bayes (BoW) | **0.9848** | 0.9400 | **0.9463** | 0.9431 |
| Logistic Regression (BoW) | **0.9848** | 0.9925 | 0.8926 | 0.9399 |
| SVM (BoW) | **0.9883** | 0.9928 | 0.9195 | **0.9547** |
| Naive Bayes (TF-IDF) | 0.9821 | 0.9850 | 0.8792 | 0.9291 |
| Logistic Regression (TF-IDF) | 0.9767 | **1.0000** | 0.8255 | 0.9044 |
| SVM (TF-IDF) | **0.9883** | 0.9928 | 0.9195 | **0.9547** |

---

## 🏆 Key Observations

- **SVM consistently performs best** across both BoW and TF-IDF.
- **TF-IDF improves precision** but can slightly reduce recall.
- **Naive Bayes is fast and strong** as a baseline model.
- Logistic Regression achieves **perfect precision (1.0)** with TF-IDF but at the cost of recall.
- Correct preprocessing and pipeline design significantly impact results.

---

## 🎯 Best Model

> ✅ **Support Vector Machine (SVM)** with both **BoW and TF-IDF**  
> **Accuracy**: ~98.83%  
> **F1-Score**: ~95.47%

---

## 🧪 How to Run

```bash
pip install pandas numpy nltk scikit-learn
```

```python
python spam_classifier.py
```

Ensure the dataset path is correct:

```
/mnt/data/SMSSpamCollection.txt
```

---

## 💡 Key Learnings

* Why **train-test split must be done before vectorization**
* Differences between **BoW and TF-IDF**
* Why empty documents break TF-IDF
* How model choice affects precision vs recall
* How to present ML results professionally

---

## 🔮 Future Enhancements

* Average Word2Vec implementation
* FastText embeddings
* Hyperparameter tuning with GridSearchCV
* Confusion matrix & ROC-AUC visualization
* Deployment using Flask or FastAPI

---

## 💼 Interview-Ready Summary

> “I built an end-to-end NLP pipeline for SMS spam detection using Bag of Words and TF-IDF features, trained Naive Bayes, Logistic Regression, and SVM models, and compared them using accuracy, precision, recall, and F1-score. SVM achieved the best overall performance.”

---

## ⭐ Final Note

This project demonstrates **strong NLP fundamentals**, **correct ML workflow**, and **clear model comparison**, making it a solid addition to any **data science or ML portfolio**.

⭐ If you found this project useful, consider starring the repository!
