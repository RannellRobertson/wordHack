import numpy as np
from keras.utils import np_utils


class Preprocessor(object):

    def __init__(self, text):
        self.text = text
        self.X = list()
        self.y = list()
        self.characters = sorted(list(set(self.text)))
        self.n_to_char = {n: char for n, char in enumerate(self.characters)}
        self.char_to_n = {char: n for n, char in enumerate(self.characters)}

        self.length = len(self.text)

        self.seq_length = 100

    def preprocessor(self):

        for i in range(0, self.length - self.seq_length, 1):
            sequence = self.text[i:i + self.seq_length]
            label = self.text[i + self.seq_length]
            self.X.append([self.char_to_n[char] for char in sequence])
            self.y.append(self.char_to_n[label])

        x_modified = np.reshape(self.X, (len(self.X), self.seq_length, 1))
        x_modified = x_modified / float(len(self.characters))
        y_modified = np_utils.to_categorical(self.y)

        return self.X, self.y, x_modified, y_modified
