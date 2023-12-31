# Install necessary libraries
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Load dataset
def load_dataset():
    df = pd.read_csv('product.csv')
    x_data = df['review'].astype(str)
    y_data = df['sentiment'].map({'positive': 1, 'negative': 0})
    english_stops = set(stopwords.words('english'))


    # PRE-PROCESS REVIEW
    x_data = x_data.apply(lambda review: re.sub(r'<.*?>', '', review))  # remove html tags
    x_data = x_data.apply(lambda review: re.sub(r'[^A-Za-z]', ' ', review))  # remove non-alphabet characters
    x_data = x_data.apply(lambda review: [w.lower() for w in review.split() if w not in english_stops])  # remove stopwords

    return x_data, y_data

# Get max review length
def get_max_length(tokenized_sequences):
    return max(map(len, tokenized_sequences))

# Architecture
def create_model(total_words, max_length):
    EMBED_DIM = 64
    LSTM_OUT = 128

    model = Sequential()
    model.add(Embedding(total_words, EMBED_DIM, input_length=max_length))
    model.add(LSTM(LSTM_OUT, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Train model
def train_model(model, x_train, y_train, x_test, y_test):
    checkpoint = ModelCheckpoint(
        'models/LSTM3.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=256, epochs=5, callbacks=[checkpoint])
    return model

# Evaluate model
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_binary))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_binary))

# Load dataset
x_data, y_data = load_dataset()

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

# Tokenize and pad sequences
token = Tokenizer()
token.fit_on_texts(x_data)

x_train_tokens = token.texts_to_sequences(x_train)
x_test_tokens = token.texts_to_sequences(x_test)

max_length = max(get_max_length(x_train_tokens), get_max_length(x_test_tokens))

x_train = pad_sequences(x_train_tokens, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test_tokens, maxlen=max_length, padding='post', truncating='post')

total_words = len(token.word_index) + 1  # Calculate total words after tokenization
e


# Create and train model
model = create_model(total_words, max_length)
model = train_model(model, x_train, y_train, x_test, y_test)

# Evaluate model
evaluate_model(model, x_test, y_test)

# Example: Load saved model and make a prediction
loaded_model = load_model('models/LSTM3.h5')

# Sample input for prediction
review_input = "Your sample review here."

# Preprocess input
review_input = re.sub(r'[^a-zA-Z\s]', '', review_input)
words = review_input.split(' ')
filtered = [w for w in words if w not in english_stops]
filtered = ' '.join(filtered)
filtered = [filtered.lower()]

# Tokenize and pad input
tokenized_input = token.texts_to_sequences(filtered)
tokenized_input = pad_sequences(tokenized_input, maxlen=max_length, padding='post', truncating='post')

# Make prediction
result = loaded_model.predict(tokenized_input)
print("Prediction Result:", result)
