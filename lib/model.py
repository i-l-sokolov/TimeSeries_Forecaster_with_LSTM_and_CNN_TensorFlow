import tensorflow as tf
from tqdm import tqdm
from math import ceil
import numpy as np


class PredictionGeneration():
    """The class for datasets generation, choosing of model hyperparameters and submission data generation"""

    def __init__(self, df_data, df_submission, window=100, out_vals=1, train_val=0.8, loss='mse', conv=False,
                 epochs=100, shuffle=False):

        """
        window - lenght of sample for prefiction next values
        out_val - the number of values to predict
        train_val - the part of train dataset compared to the whole data
        loss  - loss function during training
        conv - using convolution before LSTM part
        batch - batch size durin training
        """

        self.data = df_data
        self.sub = df_submission
        self.window = window
        self.out_vals = out_vals
        self.train_val = train_val
        self.loss = loss
        self.epochs = epochs
        self.shuffle = shuffle
        self.conv = conv
        self.train, self.val = self.dataset()
        self.model = self.get_model()
        self.best_weights = self.model.get_weights()  # will be chosen during traning

    def dataset(self):

        """
        Creation of train and validation datasets.
        """
        data_arr = self.data['sleep_hours'].values / self.data['sleep_hours'].max()

        # Making windows of data through the whole array
        X, y = [], []
        for i in range(len(data_arr) - self.window - self.out_vals + 1):
            X.append(data_arr[i:i + self.window].reshape(self.window, 1))
            y.append(data_arr[i + self.window:i + self.window + self.out_vals])
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        train_ln = round(len(X) * self.train_val)
        val_ln = len(X) - train_ln
        if self.shuffle:
            dataset = dataset.shuffle(len(X))
        train = dataset.take(train_ln).batch(train_ln)
        val = dataset.skip(train_ln).batch(val_ln)
        return train, val

    def get_model(self):
        input_model = tf.keras.Input(shape=(self.window, 1))
        if self.conv:
            def conv_dil(x, dilation):
                return tf.keras.layers.Conv1D(1, kernel_size=7, dilation_rate=dilation, padding='same',
                                              activation='relu', data_format='channels_last')(x)

            concat = tf.keras.layers.Concatenate()([conv_dil(input_model, dil) for dil in [1, 7, 31, 50]])
            out = tf.keras.layers.Dropout(0.1)(concat)
            out = tf.keras.layers.Dense(1)(out)
        else:
            out = input_model

        out = tf.keras.layers.LSTM(42)(out)

        out = tf.keras.layers.Dropout(0.1)(out)

        out = tf.keras.layers.Dense(self.out_vals)(out)

        model = tf.keras.models.Model(input_model, out)

        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        return model

    def training(self, name):
        """Training model"""
        self.val_loss = self.model.evaluate(self.val, verbose=0)
        best_epoch = 0
        for epoch in tqdm(range(self.epochs), position=0, leave=True, desc=f'training {name}'):
            history = self.model.fit(self.train, validation_data=self.val, epochs=1, verbose=0)
            if history.history['val_loss'][0] < self.val_loss:
                self.val_loss = history.history['val_loss'][0]
                self.best_weights = self.model.get_weights()
                best_epoch = epoch + 1
        self.model.set_weights(self.best_weights)
        print(f'The best results was achieved on the {best_epoch} epoch with val_loss {self.val_loss}')

    def prediction(self, name):
        y_pred = []
        data_arr = self.data['sleep_hours'].values / self.data['sleep_hours'].max()
        fst_X = data_arr[-self.window:].reshape(1, self.window, 1)
        n_preds = ceil(self.sub.shape[0] / self.out_vals)
        for _ in tqdm(range(n_preds), position=0, leave=True, desc=f'prediction {name}'):
            pred = self.model.predict(fst_X, verbose=0)
            y_pred.append(pred)
            fst_X = np.concatenate([fst_X.flatten(), pred.flatten()])[-self.window:].reshape(1, self.window, 1)
        self.sub['sleep_hours'] = np.array(y_pred).flatten()[:self.sub.shape[0]] * self.data['sleep_hours'].max()
        self.sub['val_loss'] = self.val_loss