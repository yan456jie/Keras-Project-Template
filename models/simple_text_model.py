from base.base_model import BaseModel
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model


class SimpleLstmTextModel(BaseModel):
    def __init__(self, config):
        super(SimpleLstmTextModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        # input_shape=(TIME_STEPS, INPUT_SIZE)
        self.model.add(LSTM(units = 32, input_shape=(self.config.trainer.seq_length, 1)))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def save(self):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save(self.config.callbacks.model_dir + '/my_model.h5')
        print("Model saved")

    def load(self):
        print("Loading model\n")
        self.model = load_model(self.config.callbacks.model_dir + '/my_model.h5')
        print("Model loaded")

class SimpleCnnTextModel(BaseModel):
    def __init__(self, config):
        super(SimpleCnnTextModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        max_features = self.config.trainer.embed_feature
        embed_size = 32
        maxlen = self.config.trainer.seq_length
        # Inputs
        comment_seq = Input(shape=[maxlen], name='x_seq')

        # Embeddings layers
        emb_comment = Embedding(max_features, embed_size)(comment_seq)

        # conv layers
        convs = []
        filter_sizes = [2, 3, 4]
        for fsz in filter_sizes:
            l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_comment)
            l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
            l_pool = Flatten()(l_pool)
            convs.append(l_pool)
        merge = concatenate(convs, axis=1)

        out = Dropout(0.5)(merge)
        output = Dense(32, activation='relu')(out)

        output = Dense(units=2, activation='softmax')(output)

        self.model = Model([comment_seq], output)
        #     adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])