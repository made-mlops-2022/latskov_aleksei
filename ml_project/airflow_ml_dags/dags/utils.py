from src.data_preparation import *
from src.predict import *
from src.evaluate import *

PATH = 'data/'
PATH_MODEL = 'model/'


def preparation_data():
    data = pd.read_csv(PATH + 'data_s.csv', names=range(32))

    train, test = train_test_split(data)
    split_data_on_x_y(train, 'train')
    split_data_on_x_y(test, 'test')


def train_data():
    train = pd.read_csv(PATH + 'train.csv')
    train_target = pd.read_csv(PATH + 'train_target.csv')

    del train['4']

    model = Model(train, 'adam')
    model.fit(train, train_target)
    # import os
    # PATH_MODEL = str(os.listdir()) + '/'
    # try:
    #     PATH_MODEL += str(os.listdir('model')) + '/'
    # except Exception:
    #     pass
    #
    # try:
    #     PATH_MODEL += str(os.listdir('data')) + '/'
    # except Exception:
    #     pass
    with open(PATH_MODEL + 'model.pickle', 'wb') as f:
        pickle.dump(model, f)


def predict_data():
    predict()


def evaluate_data():
    test_target = pd.read_csv(PATH + 'test_target.csv')
    test_target = test_target.values[:, :1].flatten()

    predict_target = pd.read_csv(PATH + 'predict_target.csv')
    predict_target = predict_target.values.flatten()

    score = np.mean(predict_target == test_target)
    print(f'Accuracy: {score:.6f}')
