import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# NLTK setup
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Lowercase, remove special characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords & lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load only 1000 lines
data = []
with open("train.ft.txt/train.ft.txt", encoding='utf-8') as file:   ## when we extact the file, it will be in train.ft.txt/train.ft.txt
    # Read only the first 30000 lines
    for i, line in enumerate(file):
        if i >= 30000:
            break
        label, text = line.strip().split(" ", 1)
        sentiment = 1 if label == '__label__2' else 0
        data.append((text, sentiment))

df = pd.DataFrame(data, columns=["text", "label"])
df["clean_text"] = df["text"].apply(clean_text)

# TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["clean_text"])
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Save model & vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
