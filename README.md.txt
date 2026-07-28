# 📰 Fake News Detection using Machine Learning

## Project Overview

This project detects whether a news article is **REAL** or **FAKE** using Machine Learning. A Streamlit web application allows users to enter a news article and receive an instant prediction.

---

## Dataset

- **Dataset:** Fake or Real News Dataset
- **Source:** Kaggle
- **Records:** 6,335 news articles
- **Target Column:** `label`
  - REAL
  - FAKE

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Pickle

---

## Machine Learning Model

- TF-IDF Vectorizer
- Logistic Regression Classifier

---

## Project Files

```
Fake-News-Streamlit/
│
├── app.py
├── save_model.py
├── model.pkl
├── fake_or_real_news.csv
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### Step 1: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python save_model.py
```

This will:
- Load the dataset
- Train the model
- Save the trained model as `model.pkl`

### Step 3: Run the Streamlit App

```bash
python -m streamlit run app.py
```

The application will open automatically in your web browser.

---

## How to Use

1. Open the Streamlit application.
2. Enter or paste a news article.
3. Click the **Predict** button.
4. The application will display whether the news is:
   - ✅ REAL NEWS
   - ❌ FAKE NEWS

---

## Model Performance

- **Algorithm:** Logistic Regression
- **Vectorizer:** TF-IDF
- **Accuracy:** **94.87%**

---

## Output

The application predicts whether the entered news article is **REAL** or **FAKE**.

---

## Future Improvements

- Improve prediction accuracy using advanced models.
- Deploy the application online.
- Add prediction confidence scores.
- Support multiple languages.

---



