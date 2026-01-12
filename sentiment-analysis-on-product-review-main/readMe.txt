Sentiment Analysis of Product Reviews

🧠 Project Overview:
--------------------
This project analyzes product reviews from Amazon and classifies them as **Positive 😊** or **Negative 😞** using Natural Language Processing (NLP).
The model is trained using labeled reviews from a public dataset. A simple and interactive web interface (built with Flask) allows users to input a 
review and view the predicted sentiment along with a probability chart.





📌 Dataset Information:
-----------------------
Dataset used: [Amazon Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
- Format: `__label__2` (positive), `__label__1` (negative)
- Note: After analyzing the dataset, I found that it does **not include neutral labels**. Hence, this model is designed to classify only **positive** and **negative** sentiments.


📁 Folder Structure:
--------------------
SentimentAnalysis/
│
├── templates/
│ └── index.html    # HTML template for the web interface
│
├── train.ft.txt/
│ └── train.ft.txt   # Raw dataset file (first 30,000 lines used)
│
├── app.py            # Flask web app for prediction
├── train_model.py    # Script to train and save model
├── model.pkl         # Trained Logistic Regression model
├── tfidf.pkl         # Saved TF-IDF vectorizer
├── train.csv         # (Optional) Cleaned CSV version of training data
├── test.ft.txt.bz2   # Extra dataset file (unused)
├── train.ft.txt.bz2  # Original compressed dataset
└── readMe.txt        # Project documentation (this file)
Extra have  sentiment_analysis.ipynb have all the implementation   and show all the detail 





🧰 Libraries & Environment Setup:
---------------------------------
Python 3.11 is required.

Install the necessary libraries using:

pip install -r requirements.txt
Or manually install:

pip install pandas scikit-learn nltk flask matplotlib




🔧 How to Run the Project:

Train the Model (optional if using provided model.pkl)
python train_model.py

Run the Web Application
python app.py


Open your browser and go to:
http://127.0.0.1:5000/



📥 Input & 📤 Output:

Input: A written review (plain text).

Output:
Predicted Sentiment: Positive or Negative



Sentiment Probability Bar Chart
📊 Visualization:
A bar chart shows the prediction probabilities for both classes (positive and negative).

Sentiment is displayed below the prediction button.




🔍 Testing With New Data:
You can test any custom review using the text area on the homepage of the web app UI. Enter a sentence or paragraph and click Predict Sentiment.


✨ UI Features:

Clean and responsive design
Emoji-based sentiment output
Probability bar chart for intuitive feedback
https://drive.google.com/file/d/1IhsfXe0J-kuv_SL4-b_o-IxTlDDkJuqy/view?usp=sharing




This is the classification report
	Classification Report:

               precision    recall  f1-score   support

           0       0.91      0.89      0.90    200000
           1       0.90      0.91      0.90    200000

    accuracy                           0.90    400000
   macro avg       0.90      0.90      0.90    400000
weighted avg       0.90      0.90      0.90    400000



✅ Example:
Review: "This is the best product I’ve ever bought!"
Prediction: Positive 😊
📌 Notes:
This is a binary classification task due to the dataset structure.

If a new dataset with neutral labels is available, the model can be updated for 3-class classification.

👨‍💻 Created By:
Aman Kumar Singh

