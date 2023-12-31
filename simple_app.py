from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the trained LSTM model from the H5 file
loaded_model = load_model('models/LSTM3.h5')  # Update the model filename

# Load the Tokenizer used during training
tokenizer = Tokenizer()

# Set the correct max_length based on your training data
max_length =  1481

# Define the English stop words
english_stops = set(stopwords.words('english'))

def preprocess_input(review):
    regex = re.compile(r'[^a-zA-Z\s]')
    review = regex.sub('', review)
    words = review.split(' ')
    filtered = [w for w in words if w not in english_stops]
    filtered = ' '.join(filtered)
    filtered = [filtered.lower()]
    return filtered

def predict_sentiment(review):
    preprocessed_review = preprocess_input(review)
    tokenize_words = tokenizer.texts_to_sequences(preprocessed_review)
    tokenize_words = pad_sequences(tokenize_words, maxlen=max_length, padding='post', truncating='post')
    result = loaded_model.predict(tokenize_words)
    sentiment = 'Positive' if result >= 0.5 else 'Negative'
    return sentiment, result[0][0] * 100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        sentiment, confidence = predict_sentiment(review)
        return render_template('index.html', review=review, sentiment=sentiment, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
