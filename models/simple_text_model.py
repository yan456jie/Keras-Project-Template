from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.layers import LSTM


class SimpleTextModel(BaseModel):
    def __init__(self, config):
        super(SimpleTextModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(32, input_shape=(300, 1)))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
