import numpy as np
from fetch_text import open_file
import tensorflow as tf
import keras
from preprocessing import Preprocessor
from LSTM_models import ultimate_lstm_model
from pprint import pprint


config = tf.ConfigProto(device_count={"CPU": 8})
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


text = open_file("edgar_allan_poe/edgar_allan_poe_complete_works.txt")
preprocessor = Preprocessor(text)
x_modified, y_modified = preprocessor.preprocessor()

string_mapped = preprocessor.X[99]
full_string = [preprocessor.n_to_char[value] for value in string_mapped]

model = ultimate_lstm_model(x_modified, y_modified)

model.load_weights("text_generator_700_0.2_700_0.2_700_0.2_ultimate.h5")

# generating character

for i in range(400):
    x = np.reshape(string_mapped, (1, len(string_mapped), 1))
    x = x / float(len(preprocessor.characters))
    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [preprocessor.n_to_char[value] for value in string_mapped]
    full_string.append(preprocessor.n_to_char[pred_index])

    string_mapped.append(pred_index)
    string_mapped = string_mapped[1: len(string_mapped)]

# combining the text

txt = ""

for char in full_string:
    txt += char

# write generated text to a file

with open("poem_text", "w") as f:
    f.write(txt)
    pprint(txt)
