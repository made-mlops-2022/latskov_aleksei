# module predict

from train import *


def predict():
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)

    test = pd.read_csv('test.csv')
    del test['4']
    test.to_csv('for_predict.csv')
    predict = model.predict(test)
    pd.Series(predict).to_csv(PATH + 'predict_target.csv', index=False)

    
if __name__ == '__main__':
    predict()
