import json
from keras.models import load_model

from bs4 import BeautifulSoup
import re

from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences

import numpy as np

MODEL_PATH = 'comment/classification/saved_data/rnn_model.h5'
TOKENIZER_PATH = 'comment/classification/saved_data/tokenizer.json'
MAX_LEN = 300

with open(TOKENIZER_PATH) as f:
    TOKENIZER = tokenizer_from_json(json.load(f))

model = load_model(MODEL_PATH)


# Удаление html-тегов
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# Удаление квадратных скобок
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


# Удаление специальных символов
def remove_special_characters(text):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


def prepare_data(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_special_characters(text)
    return text


def get_prediction(text):
    text = prepare_data(text)
    x = TOKENIZER.texts_to_sequences([text])
    print(x)
    x = pad_sequences(x, MAX_LEN)
    result = model.predict(x)
    mark = int(np.round(result * 10))
    if mark >= 5:
        sentiment = 'положительный'
    else:
        sentiment = 'отрицательный'
    return sentiment, mark


if __name__ == "__main__":
    print(get_prediction('The film is awesome'))
