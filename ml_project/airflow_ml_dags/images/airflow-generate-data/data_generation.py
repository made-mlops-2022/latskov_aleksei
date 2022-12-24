import pandas as pd
import numpy as np


def generate_data():
    columns = [str(i) for i in range(2, 32)]

    df = pd.DataFrame(columns=columns)
    test = pd.read_csv('test.csv', index_col=False)
    for column in columns:
        df[column] = np.random.choice(test[column].values, 10)

    output_dir = '/data'
    df.to_csv(f"{output_dir}/test.csv", index=False)


if __name__=="__main__":
    generate_data()
