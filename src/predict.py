# module predict

import pickle
import pandas as pd


def main():
    try:
        with open('model.pickle', 'rb') as f:
            model = pickle.load(f)
    except Exception as error:
        print(error)

    test = pd.read_csv('test.csv')
    del test['4']

    predict = model.predict(test)
    pd.Series(predict).to_csv('predict_target.csv', index=False)

    
if __name__ == '__main__':
    main()
