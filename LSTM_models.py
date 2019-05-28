from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM


# A model that has three LSTM layers, each with 700 units
# trained for 100 epochs.
def lstm_model(X, y):
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
