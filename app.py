from flask import Flask, request, render_template, redirect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
import ssl
import psycopg2
import os
from dotenv import load_dotenv

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
load_dotenv()

app = Flask(__name__)

stop_words = stopwords.words('english')
sentiment_analyzer = SentimentIntensityAnalyzer()

INSERT_INPUTS = "INSERT INTO inputs (query) VALUES (%s);"
RETRIEVE_INPUTS = "SELECT query FROM inputs ORDER BY id DESC LIMIT 5;"


def get_db_connection():
    user = os.environ.get("PG_USER")
    password = os.environ.get("PG_PASSWORD")
    host = os.environ.get("PG_HOST")
    port = os.environ.get("PG_PORT")
    database = os.environ.get("PG_DATABASE")
    conn = psycopg2.connect(user=user, password=password, host=host, port=port, database=database)
    return conn

@app.route('/')
def sentiment_form():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(RETRIEVE_INPUTS)
    inputs = cursor.fetchall()
    print(inputs)
    cursor.close()
    conn.close()
    return render_template('form.html', inputs=inputs)

@app.route('/', methods=['POST'])
def analyze_sentiment():
    user_input = request.form['text'].strip()
    if(user_input == ""):
        return redirect("/");
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(INSERT_INPUTS, (user_input,))
    cursor.execute(RETRIEVE_INPUTS)
    inputs = cursor.fetchall()
    cursor.close()
    conn.commit()
    conn.close()
    user_input_lower = user_input.lower()
    text_without_digits = ''.join(c for c in user_input_lower if not c.isdigit())
    processed_text = ' '.join([word for word in text_without_digits.split() if word not in stop_words])
    sentiment_scores = sentiment_analyzer.polarity_scores(text=processed_text)
    compound_score = round((1 + sentiment_scores['compound']) / 2, 2)
    return render_template('form.html', compound=compound_score, text=user_input, positivity=sentiment_scores['pos'], negativity=sentiment_scores['neg'], neutrality=sentiment_scores['neu'], inputs=inputs)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
