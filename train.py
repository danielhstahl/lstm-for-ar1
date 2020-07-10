from simulate_ts import get_ar_ts
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import math
import itertools


def create_lookback(ts_data: List[float], look_back: int) -> List[List[float]]:
    return [ts_data[i : look_back + i] for i in range((len(ts_data) + 1 - look_back))]


def train_model(
    look_back: int, ts_data: List[float], split_test: float
) -> Tuple[List[float], List[float]]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    ts_data_scaled = scaler.fit_transform(ts_data)
    ts_data_lookback = np.array(create_lookback(ts_data_scaled, look_back))
    print(ts_data_lookback.shape)
    ts_data_lookback = np.reshape(
        ts_data_lookback, (ts_data_lookback.shape[0], 1, look_back)
    )
    model = Sequential()
    model.add(LSTM(25, input_shape=(1, look_back)))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    n = len(ts_data_lookback)
    num_train = int(split_test * n)
    num_test = n - num_train
    ts_data_train = ts_data_lookback[:num_train]
    ts_data_test = ts_data_lookback[num_train:]
    ts_data_actual = ts_data[
        (num_train + 1) :
    ]  # need "+1" since test is fully new, including "base" element.
    # "online mode" with batch_size=1.  Not sure how batch_size other than 1 would work with timeseries
    model.fit(
        ts_data_train[:-1], ts_data_train[1:], epochs=1000, batch_size=1, verbose=1,
    )
    print(ts_data_lookback.shape)
    print(ts_data_train.shape)
    print(ts_data_test.shape)
    print(len(ts_data_actual))
    print(num_train)
    return scaler.inverse_transform(model.predict(ts_data_test[:-1])), ts_data_actual

    # return model


def main():
    alpha = 0.2

    beta = 0.7
    sigma = 0.1
    x0 = 10
    ts1 = [[x] for x in get_ar_ts(alpha, beta, sigma, x0, 100)]  # single feature

    ts2 = [[x] for x in get_ar_ts(alpha, beta, sigma, x0, 1000)]

    ts3 = [[x] for x in get_ar_ts(alpha, beta, sigma, x0, 10000)]

    look_back = [1, 10]  ## lookback>1 shouldn't make a difference, since DGP is ar(1)
    series_lookback = itertools.product(
        [(ts1, "ts1"), (ts2, "ts2"), (ts3, "ts3")], look_back
    )
    train_data_perc = 0.9
    results = [
        (*train_model(look_back, ts[0], train_data_perc), ts[1], look_back)
        for ts, look_back in series_lookback
    ]

    for predict, actual, tsname, look_back in results:
        print(
            f"time series {tsname} with look back {look_back}: {math.sqrt(mean_squared_error(predict, actual))}"
        )


if __name__ == "__main__":
    main()
