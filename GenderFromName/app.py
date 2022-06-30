from flask import Flask, render_template, request
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)


@app.route('/', methods=['Get'])
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        result = request.form['Name']
        print(result)
        save_path = './Model'
        loaded_model = tf.keras.models.load_model(save_path)
        tokenizer = Tokenizer(num_words=100, lower=True,
                              char_level=True, oov_token='UNK')
        tokenizer.fit_on_texts([result])
        word_represent = tokenizer.texts_to_sequences([result])
        x_pred = pad_sequences(word_represent, padding='post', maxlen=20)
        gender_probability = loaded_model.predict(x_pred)
        print(gender_probability)
        return render_template("index.html", result='Female' if gender_probability[0][0] > 0.5 else 'Male')


if __name__ == '__main__':
    app.run(port=3000, debug=True)
