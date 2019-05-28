from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM


def lstm_baseline_model(X, y):
    model = Sequential()
    model.add(LSTM(400,  input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(400))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.fit(X, y, epochs=1, batch_size=100)
    model.save("lstm_baseline_model.h5")
    model.save_weights("text_generator_400_0.2_400_0.2_baseline.h5")
    return model


def lstm_more_trained_model(X, y):
    model = Sequential()
    model.add(LSTM(400,  input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(400))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    # Model with 100 epochs
    model.fit(X, y, epochs=100, batch_size=100)
    model.save("lstm_more_trained_model.h5")
    model.save_weights("text_generator_400_0.2_400_0.2_more_trained.h5")
    return model


# A deeper model with 3 LSTM layers with 400 units
# and 3 dropout laters of 0.2
def lstm_deeper_model(X, y):
    model = Sequential()
    model.add(LSTM(400, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(400))
    model.add(Dropout(0.2))
    model.add(LSTM(400))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.fit(X, y, epochs=100, batch_size=100)
    model.save("lstm_deeper_model.h5")
    model.save_weights("text_generator_400_0.2_400_0.2_400_0.2_deeper.h5")
    return model


# A model with 700 units on each two LSTM layers
def lstm_wider_model(X, y):
    model = Sequential()
    model.add(LSTM(700, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(700))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.fit(X, y, epochs=100, batch_size=100)
    model.save("lstm_wider_model.h5")
    model.save_weights("text_generator_700_0.2_700_0.2_wider.h5")
    return model


# A model that incorporates features of previous
# models. Three LSTM layers, each with 700 units
# trained for 100 epochs.
def ultimate_lstm_model(X, y):
    model = Sequential()
    model.add(LSTM(700, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(700, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(700))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.fit(X, y, epochs=100, batch_size=100)
    model.save("models/lstm_model.h5")
    model.save_weights("models/text_generator_700_0.2_700_0.2_700_0.2.h5")
    return model
