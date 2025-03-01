import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[^a-zA-Z ]", "", text.lower())  # Keep only letters and spaces
    text = " ".join(text.split())  # Remove extra spaces
    return text


# Load datasets
fake_data = pd.read_csv("C:/Users/Lenovo/Downloads/Fake.csv")
true_data = pd.read_csv("C:/Users/Lenovo/Downloads/True.csv")

# Add labels: 0 for Fake, 1 for Real
fake_data["label"] = 0
true_data["label"] = 1

# Keep only necessary columns
fake_data = fake_data[["title", "text", "label"]]
true_data = true_data[["title", "text", "label"]]

# Merge datasets
data = pd.concat([fake_data, true_data], ignore_index=True)

# Fill NaN values and combine title + text
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")
data["combined_text"] = data["title"] + " " + data["text"]
data["combined_text"] = data["combined_text"].apply(preprocess_text)

# Remove empty texts
data = data[data["combined_text"].str.strip() != ""]

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(data["combined_text"])
Y = data["label"].astype(int)

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate model
Y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


# Function to predict user input
def predict_news(news_text):
    processed_text = preprocess_text(news_text)
    transformed_text = vectorizer.transform([processed_text])
    prediction = model.predict(transformed_text)[0]
    return "Real News" if prediction == 1 else "Fake News"


# Get user input
news_input = input("Enter news text: ")
result = predict_news(news_input)
print(f"Prediction: {result}")
