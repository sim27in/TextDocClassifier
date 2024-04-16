import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
import tensorflow as tf
import tensorflow_hub as hub
import joblib

download("punkt")
download("stopwords")
download("omw-1.4")
download("wordnet")
download("wordnet2022")


class TextClassifier:
    def __init__(self, model_path, encoder_path):
        self.model = tf.keras.models.load_model(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub("[\d\W_]+", " ", text)
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in nltk.word_tokenize(text)
            if token not in self.stop_words
        ]

        return " ".join(tokens)

    def embed_text(self, text):
        embedded_text = self.embed([text])[0]

        return embedded_text.numpy()

    def predict(self, text):
        preprocessed_text = self.preprocess_text(text)
        embedded_text = self.embed_text(preprocessed_text)
        embedded_text = np.array(embedded_text.tolist())
        embedded_text = embedded_text.reshape((1, 1, embedded_text.shape[0]))
        predictions = self.model.predict(embedded_text)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = self.label_encoder.inverse_transform([predicted_class_index])[0]
        probabilities = predictions[0].tolist()

        return predicted_class, probabilities