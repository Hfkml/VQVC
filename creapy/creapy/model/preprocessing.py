from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from ..utils import get_config


def impute(X_train: pd.DataFrame,
           X_test: Optional[pd.DataFrame] = None,
           return_imputer: bool = False):

    config_ = get_config()["MODEL"]["PREPROCESSING"]
    impute_strategy = config_["impute_strategy"]
    _imputer = SimpleImputer(strategy=impute_strategy)
    if X_test is None:
        res = pd.DataFrame(_imputer.fit_transform(
            X_train), columns=X_train.columns)
    else:
        X_train = pd.DataFrame(_imputer.fit_transform(
            X_train), columns=X_train.columns)
        X_test = pd.DataFrame(_imputer.transform(X_test),
                              columns=X_train.columns)
        res = X_train, X_test
    return (res, _imputer) if return_imputer is True else res


def split_data(X: pd.DataFrame, y: pd.Series):
    config_ = get_config()["MODEL"]["PREPROCESSING"]
    impute_strategy = config_["impute_strategy"]
    test_size = config_["test_size"]
    if impute_strategy is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=config_["random_state"])
        X_train, X_test = impute(X_train, X_test)
    else:
        _tmp = X.copy()
        _tmp["label"] = y
        _tmp.dropna(inplace=True)
        _X_drop = _tmp.drop(["label"], axis=1)
        _y_drop = _tmp["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            _X_drop, _y_drop, test_size=test_size, random_state=config_["random_state"])
        del _tmp, _X_drop, _y_drop

    return X_train, X_test, y_train, y_test


def buffer(x, sr, opt: str = "nodelay", window=None):
    """
    Buffer signal vector into matrix of data frames

    x: Signal
    N: block size
    OL: overlap
    window: predefined window

    * works only for 1D Signals!
    """

    config_ = get_config()["USER"]
    N = int(config_["block_size"] * sr)
    R = int(config_["hop_size"] * sr)

    if window is None:
        window = np.ones(N)

    assert len(window) == N, "windowlength does not match blocksize N"
    n = len(x)
    OL = int(N - R)
    if opt == 'nodelay':
        assert n >= OL, "in 'nodelay' mode, len(x) must be OL or longer"
        n_seg = int(np.ceil((n - N) / R + 1))
        # print('num_seg:', n_seg)
    else:
        n_seg = int(np.ceil(n / R))
        x = np.concatenate([np.zeros(OL), x.squeeze()])

    res = np.zeros((N, n_seg))
    for i in range(n_seg - 1):
        res[:, i] = x[i * R: i * R + N] * window

    i = n_seg - 1
    # last block
    nLast = x[i * R:].size
    res[range(nLast), n_seg - 1] = x[i * R:]
    return res
