



from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# Download NLTK assets if not present
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Preprocessing Function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    cleaned_text = preprocess(input_text)
    vect_text = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(vect_text)[0]
    probas = model.predict_proba(vect_text)[0]
    
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"

    return render_template("index.html", prediction_text=sentiment, input_text=input_text,
                           probabilities=list(map(float, probas)))

if __name__ == "__main__":
    app.run(debug=True)
