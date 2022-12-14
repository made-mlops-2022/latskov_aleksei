# module predict

import pickle
import logging.config
import pandas as pd

import matplotlib.pyplot as plt

from config import OPTIMIZER_CONFIG, OPTIMIZER_ADAM_CONFIG
from net import *

PATH = '../data/'
PATH_LOGS = '../logs/'
PATH_MODEL = '../model/'


log_conf = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "%(asctime)s\t%(levelname)s\t%(message)s",
        },
    },
    "handlers": {
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "filename": f"{PATH_LOGS}file_handler.log",
            "formatter": "simple",
            },
        "stream_handler": {
            "level": "DEBUG",
            "formatter": "simple",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {
            "level": "INFO",
            "handlers": ["file_handler", "stream_handler"],
        },
    },
}


class Model:
    def __init__(self, frame, optim='sgd'):
        self.optim = optim
        self.mean = frame.values.mean(axis=0)
        self.std = frame.values.std(axis=0)
        self.n_in = len(frame.columns)
        self.model = get_model(self.n_in, 2, 3)
    
    def std_scaler(self, x):
        return (x - self.mean) / self.std
        
    def train(self, X, y, n_epoch=50, batch_size=1000):
        criterion = ClassNLLCriterion()
        optimizer_state = {}

        loss_history = []
        accuracy_train = []

        for i in range(n_epoch):
            for x_batch, y_batch in get_batches((X, y), batch_size):
                self.model.train()
                self.model.zeroGradParameters()
                
                # Forward
                predictions = self.predict_proba(x_batch)
                loss = criterion.forward(predictions, y_batch)
                if np.isnan(loss):
                    break
                
                # Backward
                dp = criterion.backward(predictions, y_batch)
                self.model.backward(x_batch, dp)

                # Update weights
                if self.optim == 'sgd':
                    simple_sgd(
                        self.model.getParameters(),
                        self.model.getGradParameters(),
                        OPTIMIZER_CONFIG,
                        optimizer_state,
                )
                if self.optim == 'adam':
                    adam_optimizer(self.model.getParameters(),
                                   self.model.getGradParameters(),
                                   OPTIMIZER_ADAM_CONFIG,
                                   optimizer_state)

                loss_history.append(loss)

            if np.isnan(loss):
                break

            self.model.evaluate()
            y_train_eval = y[:, :1].reshape(-1)
            y_pred = self.predict_proba(X).argmin(axis=-1)
            accuracy_train.append(self.evaluate(y_pred, y_train_eval))
            logger.info(f"Current loss: {loss:.6f}")

        # Visualize
        plt.figure(figsize=(8, 6))
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(accuracy_train, label='Train')
        plt.legend(loc='best')
        plt.show()

        logger.info(f'Accuracy train: {accuracy_train[-1]:.6f}')
        
    def evaluate(self, y_pred, y_true):
        return np.mean(y_pred == y_true)
        
    def fit(self, frame, y):
        X = self.std_scaler(frame)
        self.train(X.values, y.values)
    
    def predict_proba(self, X):
        return self.model.forward(X)
    
    def predict(self, frame):
        X = self.std_scaler(frame)
        return self.predict_proba(X.values).argmin(axis=-1)
    

def main():    
    train = pd.read_csv(PATH + 'train.csv')
    train_target = pd.read_csv(PATH + 'train_target.csv')

    del train['4']

    model = Model(train, 'adam')
    model.fit(train, train_target)

    with open(PATH_MODEL + 'model.pickle', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    logging.config.dictConfig(log_conf)
    logger = logging.getLogger()
    main()
