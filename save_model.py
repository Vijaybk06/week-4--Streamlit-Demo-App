import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\Fake-News-Streamlit\fake_or_real_news.csv")

# Drop unwanted column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Combine title and text
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

X = df["content"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("="*50)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("="*50)
print(classification_report(y_test, y_pred))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Saved Successfully!")

# Print one REAL sample
print("\n================ REAL SAMPLE ================\n")
real = df[df["label"]=="REAL"].sample(1, random_state=42)
print(real.iloc[0]["title"])
print()
print(real.iloc[0]["text"])

# Print one FAKE sample
print("\n================ FAKE SAMPLE ================\n")
fake = df[df["label"]=="FAKE"].sample(1, random_state=42)
print(fake.iloc[0]["title"])
print()
print(fake.iloc[0]["text"])