import pandas as pd

from features import *
from helpers import *
import argparse

from helpers import load_model


def main():
    parser = argparse.ArgumentParser(description='Process accelerometer data.')

    parser.add_argument('model', action='store',
                        help='name of model to test data e.g., rf, cnn')

    parser.add_argument('input_file', action='store',
                        help='input file to pass to the model')
    parser.add_argument('output_file', action='store',
                        help='path to save the output of the model')

    args = parser.parse_args()
    print(args.model)

    model = None
    sampling_rate = 100
    window_size = 1
    window_size = window_size * sampling_rate

    try:
        model = load_model(args.model)
    except OSError:
        print("Couldn't load the model")

    data = pd.read_csv(args.input_file, header=None)
    acc_data = data.filter(items=[1, 2, 3], axis=1)
    eng_windows = rf_windowing(acc_data.to_numpy(dtype=np.float32))

    labels = None
    if model is not None:
        labels = model.predict(eng_windows)

    predictions = []
    for label in labels:
        label_name = pamap2_updated_labels[label]
        predictions.append([label_name]*window_size)

    predictions = np.array(predictions).reshape(-1)
    result = data.join(pd.DataFrame(predictions), how='outer',lsuffix='_caller', rsuffix='_other')
    result = result.dropna()
    result.to_csv(args.output_file)

    print('All Done')
    #print(data)


if __name__ == "__main__":
    main()