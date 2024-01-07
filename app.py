# Flask app
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPool1D, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)

# Load the model and tokenizer
model = tf.keras.models.load_model("/workspaces/Superhero_name_generator/models/superhero_model.h5")
tokenizer_path = '/workspaces/Superhero_name_generator/models/tokenizer.pkl'
with open(tokenizer_path, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

@app.route('/')
def home():
    return render_template('index.html')# Flask automatically looks for HTML files in a folder named "templates"

@app.route("/generate_superhero", methods=["POST"])
def generate_superhero():
    data = request.form.get('seed_name')
    if not data.strip():
        return render_template("index.html")

    generated_name = gen_names(data)
    return render_template("index.html", generated_name=generated_name, seed_name=data)

def gen_names(seed):
    seq = [tokenizer.texts_to_sequences(c)[0][0] for c in seed]
    padded = pad_sequences([seq], padding='pre', maxlen=model.input_shape[1])

    generated_name = seed
    for i in range(40):
        pred = model.predict(padded)[0]
        pred_char = tokenizer.index_word[tf.argmax(pred).numpy()]
        generated_name += pred_char

        if pred_char == '\t':
            break

        seq = [tokenizer.texts_to_sequences(c)[0][0] for c in generated_name]
        padded = pad_sequences([seq], padding='pre', maxlen=model.input_shape[1])

    return generated_name

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)