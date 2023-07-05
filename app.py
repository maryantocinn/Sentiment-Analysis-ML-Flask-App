from flask import Flask, request, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

app = Flask(__name__)
stop_words = stopwords.words('english')
sentiment_analyzer = SentimentIntensityAnalyzer()

@app.route('/')
def sentiment_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def analyze_sentiment():
    user_input = request.form['text']
    user_input_lower = user_input.lower()
    text_without_digits = ''.join(c for c in user_input_lower if not c.isdigit())
    processed_text = ' '.join([word for word in text_without_digits.split() if word not in stop_words])
    sentiment_scores = sentiment_analyzer.polarity_scores(text=processed_text)
    compound_score = round((1 + sentiment_scores['compound']) / 2, 2)
    return render_template('form.html', compound=compound_score, text=user_input, positivity=sentiment_scores['pos'], negativity=sentiment_scores['neg'], neutrality=sentiment_scores['neu'])

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
