# module predict

from train import *


def predict():
    with open(PATH_MODEL + 'model.pickle', 'rb') as f:
        model = pickle.load(f)
    print(PATH_MODEL)
    test = pd.read_csv(PATH + 'test.csv')
    del test['4']
    test.to_csv(PATH + 'for_predict.csv')
    predict = model.predict(test)
    pd.Series(predict).to_csv(PATH + 'predict_target.csv', index=False)


predict()
