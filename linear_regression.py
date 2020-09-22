# Exercise 1
# Simple Linear Regression with Synthetic Data

import pandas
import tensorflow
from matplotlib import pyplot as plt

class LRModel:
    def __init__(self, learning_rate):
        self.model = self._build_model(learning_rate)

    def _build_model(self, learning_rate):
        model = tensorflow.keras.models.Sequential()
        model.add(tensorflow.keras.layers.Dense(units = 1, input_shape = (1, )))
        model.compile(optimizer = tensorflow.keras.optimizers.RMSprop(lr = learning_rate),
                      loss = 'mean_squared_error',
                      metrics = [tensorflow.keras.metrics.RootMeanSquaredError()])

        return model

    def train(self, data, epochs, batch_size):
        self.features, self.labels = zip(*data) 
        history = self.model.fit(x = self.features,
                                 y = self.labels,
                                 batch_size = batch_size,
                                 epochs = epochs)

        self.epochs = history.epoch
        self.rms_error = pandas.DataFrame(history.history)['root_mean_squared_error']

    def _plot_model(self):
        plt.xlabel('feature')
        plt.ylabel('label')

        plt.scatter(self.features, self.labels)

        trained_weight, trained_bias = self.model.get_weights()

        x = [0, self.features[-1]]
        y = [trained_bias, trained_bias + trained_weight * x[1]]

        model = '%d + %dx' % (trained_bias, trained_weight)

        plt.plot(x, y, c='r', label=model)
        plt.legend()
        plt.show(block=False)

    def _plot_loss(self):
        plt.figure()

        plt.xlabel('Epoch')
        plt.ylabel('RMSE')

        plt.plot(self.epochs, self.rms_error, label='Loss')
        plt.legend()
        plt.ylim([self.rms_error.min() * 0.97, self.rms_error.max()])
        plt.show()

    def plot(self):
        self._plot_model()
        self._plot_loss()

if __name__ == '__main__':
    data = [(1.0, 5.0),
            (2.0, 8.8),
            (3.0, 9.6),
            (4.0, 14.2),
            (5.0, 18.8),
            (6.0, 19.5),
            (7.0, 21.4),
            (8.0, 26.8),
            (9.0, 28.9),
            (10.0, 32.0),
            (11.0, 33.8),
            (12.0, 38.2)]

    while True:
        lr = float(input('Enter learning rate: '))
        epoch = int(input('Enter epoch: '))
        batch = int(input('Enter batch: '))

        lr_model = LRModel(lr)
        lr_model.train(data, epoch, batch)

        lr_model.plot()