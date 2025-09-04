from __future__ import annotations

import time
import warnings
from functools import partial
from pathlib import Path
# from threading import Thread
from typing import Optional
import os
import numpy as np
import pandas as pd
from rich.progress import track
from scipy.signal.windows import hann

from ..feature_extraction import WINDOW_MAPPING
from ..feature_extraction.feature_extraction import (
    blockwise_feature_calculation, calculate_features, get_feature_list)
from ..utils.config import get_config
from ..utils.helpers import (ThreadWithReturnValue, get_creak_intervals,
                             get_root, get_time_vector, intervals_to_textgrid,
                             intervals_to_csv)
from ..utils.read_wav import read_wav
from .model import load_model
from .postprocessing import thresholding
from .preprocessing import buffer


def process_file(audio_path,
                 textgrid_path: Optional[str] = None,
                 csv_folder_path: Optional[str] = None,
                 gender_model: Optional[str] = None):
    _config = get_config()
    verbosity = _config['USER']['verbose']
    start, end = _config['USER']['audio_start'], _config['USER']['audio_end']
    data, sr = read_wav(audio_path, start=start, end=end)

    w = hann(int(_config['USER']["block_size"] * sr))
    creak_data_buff = buffer(data, sr, window=w)
    PREPROCESSING_FEATURES = [key for key, val in _config['MODEL']
                              ['PREPROCESSING']['UNVOICED_EXCLUSION'].items() if val is True]
    t = time.time()
    threads = [ThreadWithReturnValue(target=blockwise_feature_calculation, args=[
                                     creak_data_buff.T.copy(), sr, feature, _config]) for feature in PREPROCESSING_FEATURES]
    for thread in threads:
        thread.start()
    elimination_chunks = np.array([thread.join() for thread in threads]).T

    preprocessing_values = _config["MODEL"]['PREPROCESSING']['UNVOICED_EXCLUSION']["VALUES"]
    preprocessing_values['ZCR']['threshold'] = _config['USER']['zcr_threshold']
    preprocessing_values['STE']['threshold'] = _config['USER']['ste_threshold']
    
    thresholds = np.array(
        [thresholding(series=chunk, **preprocessing_values
                      [feature.upper()]) for chunk, feature in zip(elimination_chunks.T, PREPROCESSING_FEATURES)]
    )

    included_indices = thresholds.sum(axis=0) == 0

    features_for_classification = _config["MODEL"]["FEATURES"]["for_classification"]

    t = time.time()
    threads = [ThreadWithReturnValue(target=blockwise_feature_calculation, args=[
                                     creak_data_buff.T[included_indices].copy(), sr, feature, _config.copy()]) for feature in features_for_classification]
    for thread in threads:
        thread.start()
    _X_test = np.array([thread.join() for thread in threads]).T
    # print("time ellapsed:", time.time() - t)

    _X_test = pd.DataFrame(_X_test, columns=features_for_classification,
                           index=np.argwhere(included_indices).ravel())

    X_test = pd.concat((pd.DataFrame(elimination_chunks,
                       columns=PREPROCESSING_FEATURES), _X_test), axis=1)
    y_pred = np.zeros((creak_data_buff.shape[1]))
    if not any(included_indices):
        warnings.warn("Did not make classification. Consider setting new values "
                      "for Zero-Crossing-Rate (zcr) or Short-Term-Energy (ste).")
        return X_test, y_pred

    # Load model
    if gender_model is not None:
        gender_model = gender_model.lower().strip()
        if gender_model not in ("male", "female", "all"):
            raise ValueError(
                f'Gender must be \"male\", \"female\", \"all\" or None is {gender_model}')

    else:
        gender_model = _config['USER']['gender_model']
    model_path = get_root() / _config["MODEL"]["model_location"]
    model_path = (model_path.parent /
                  (f"{model_path.stem}_{gender_model.upper()}")).with_suffix(".csv")
    model = load_model(model_path)

    y_pred[included_indices] = model.predict(_X_test)

    if textgrid_path is not None:
        tier_name = _config['USER']['tier_name']
        filename_extension = _config['USER']['filename_extension']
        
        if filename_extension:
            _textgrid_path = Path(textgrid_path)
            new_filename = _textgrid_path.stem + filename_extension + _textgrid_path.suffix
            result_path = str(_textgrid_path.parent / new_filename)
        else:
            result_path = textgrid_path

        intervals = get_creak_intervals(
            y_pred, get_time_vector(y_pred, sr, start), tgt_intervals=True)
        intervals_to_textgrid(
            intervals=intervals,
            textgrid_path=textgrid_path,
            # result_path=textgrid_path if textgrid_dst is None else textgrid_dst,
            result_path=result_path,
            tier_name=tier_name,
            verbose=verbosity
        )

    if csv_folder_path is not None:
        intervals = get_creak_intervals(
            y_pred, get_time_vector(y_pred, sr, start), tgt_intervals=True)
        intervals_to_csv(
            intervals=intervals,
            csv_dst=csv_folder_path
        )
        pass
    return X_test, y_pred, sr


def process_folder(audio_directory: Optional[str] = None,
                   textgrid_directory: Optional[str] = None,
                   csv_directory: Optional[str] = None):
    _config = get_config()

    if audio_directory is None:
        audio_directory = _config['USER']['audio_directory']
    if not os.path.isdir(audio_directory):
        raise ValueError(
            f"Invalid path, given audio-directory \"{audio_directory}\" is not a directory.")
    if textgrid_directory is None:
        textgrid_directory = _config['USER']['textgrid_directory']
    if textgrid_directory:
        if not os.path.isdir(textgrid_directory):
            raise ValueError(
                f"Invalid path, given textgrid-directory \"{textgrid_directory}\" is not a directory.")

    if csv_directory is None:
        csv_directory = _config['USER']['csv_directory']
    if csv_directory:
        if not os.path.isdir(csv_directory):
            raise ValueError(
                f"Invalid path, given csv_directory \"{csv_directory}\" is not a directory.")

    audio_suffix = _config['USER']['audio_suffix']
    wav_files = list(Path(audio_directory).glob(f'**/*{audio_suffix}'))

    if textgrid_directory:
        wav_tg_map = dict()
        textgrid_path = Path(textgrid_directory)
        textgrid_suffix = _config['USER']['textgrid_suffix']
        for wav_file in wav_files:
            wav_tg_map[wav_file] = (
                textgrid_path / wav_file.stem).with_suffix(textgrid_suffix)

    # for wav_file in track(wav_files, "Processing folder..."):
    for wav_file in wav_files:
        process_file(
            wav_file,
            textgrid_path=wav_tg_map[wav_file] if textgrid_directory else None,
            csv_folder_path=csv_directory if csv_directory else None
        )
