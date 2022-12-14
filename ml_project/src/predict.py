# module predict

from train import *


def predict():
    try:
        with open(PATH_MODEL + 'model.pickle', 'rb') as f:
            model = pickle.load(f)
    except Exception as error:
        print(error)

    test = pd.read_csv(PATH + 'test.csv')
    del test['4']

    predict = model.predict(test)
    pd.Series(predict).to_csv(PATH + 'predict_target.csv', index=False)
    print('predict done')

    
if __name__ == '__main__':
    predict()
