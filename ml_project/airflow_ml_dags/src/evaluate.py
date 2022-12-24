# module evaluate

import pandas as pd
import numpy as np

PATH = 'data/'


def main():
    test_target = pd.read_csv(PATH + 'test_target.csv')
    test_target = test_target.values[:, :1].flatten()

    predict_target = pd.read_csv(PATH + 'predict_target.csv')
    predict_target = predict_target.values.flatten()

    score = np.mean(predict_target == test_target)
    print(f'Accuracy: {score:.6f}')
    

if __name__ == '__main__':
    main()
