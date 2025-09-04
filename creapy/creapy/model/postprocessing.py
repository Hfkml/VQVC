from __future__ import annotations

import numpy as np
import operator as operator_

def thresholding(
                 series: np.ndarray, 
                 threshold: float,
                 y: np.ndarray = None, 
                 replace_value: float = 0.0,
                 operator: str = ">=",
                 normalize: bool = False):
    
    _operators = {
        ">": operator_.gt,
        ">=": operator_.ge,
        "<": operator_.lt,
        "<=": operator_.le
    }
    
    # assert y.shape == series.shape
    if normalize is True: series /= max(series)
    if y is not None:
        y[_operators[operator](series, threshold)] = replace_value
        return y
    else:
        return _operators[operator](series, threshold)
    

def moving_average(series: np.ndarray, N: int = 10, mode: str = "same"):
    if len(series) < N: 
        return series
    return np.convolve(series, np.ones((N)) / N, mode=mode)

    