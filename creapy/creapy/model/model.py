from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer

from ..utils import get_config, get_root
from .postprocessing import moving_average
from .preprocessing import impute


class NotFittedError(Exception):
    pass

class Model:
    """The Model for creaky voice classification.
    """
    def __init__(self):
        self._config = get_config()["MODEL"]
        self._X_train: pd.DataFrame
        self._y_train: pd.Series
        self._imputer: SimpleImputer
        self._features = self._config["FEATURES"]["for_classification"]
        self._fitted = False
        _clf = self._config["CLASSIFIER"]["clf"]
        self._clf = clfs[_clf](
            **self._config["CLASSIFIER"]["VALUES"][_clf.upper()]["kwargs"])

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """Function to fit the model with training data.

        Args:
            X_train (pd.DataFrame): Features of training data.
            y_train (pd.Dataframe): Targets of training data (creak, no-creak).
        """
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.to_numpy()
        if self._config["PREPROCESSING"]["impute_at_fit"] is True:
            self._X_train, self._imputer = impute(
                X_train=X_train.loc[:, self._features], return_imputer=True)
        else:
            self._X_train = X_train
        self._y_train = pd.Series(y_train, name=self._config["target_label"])
        self._clf.fit(
            self._X_train.loc[:, self._features], self._y_train.ravel())
        self._fitted = True

    def predict(self, X_test: pd.DataFrame, predict_proba: bool=None) -> np.ndarray:
        """Predicts the given features. 

        Args:
            X_test (pd.DataFrame): Features to be predicted.
            predict_proba (bool, optional): If `True` the likelihood to be creak will be returned, else the predicted target. 
            Defaults to None.

        Returns:
            np.ndarray: Predicted targets, or probability of creak.
        """
        self._config = get_config()["MODEL"]
        if predict_proba is not None:
            assert isinstance(predict_proba, bool)
        else:
            predict_proba = self._config["CLASSIFIER"]["predict_proba"]
        if hasattr(self, "_imputer"):
            X_test = pd.DataFrame(self._imputer.transform(
                X_test.loc[:, self._features]), columns=self._X_train.columns, index=X_test.index)
        if predict_proba is True:
            _target_index = np.argwhere(
                self._clf.classes_ == self._config["CLASSIFIER"]["target_name"]).item()
            y_pred = self._clf.predict_proba(X_test[self._features])[
                :, _target_index].flatten()
            if self._config["POSTPROCESSING"]["MAVG"]["mavg"] is True:
                length, mode = map(
                    self._config["POSTPROCESSING"]["MAVG"]["VALUES"].get, ("length", "mode"))
                y_pred = moving_average(y_pred, length, mode)
        else:
            y_pred = self._clf.predict(X_test[self._features])

        return y_pred

    def save(self, filepath: str = None):
        """Saves a fitted model to the given location as csv file

        Args:
            filepath (str, optional): Destination path of saved model. Defaults to None.

        Raises:
            NotFittedError: If the model is not yet fitted it can not be saved.
        """
        if self._fitted is False:
            raise NotFittedError(
                "Can't save model because it is not fitted yet")
        _config = get_config()
        if filepath is None:
            # filepath without suffix!
            filepath = get_root() / _config["MODEL"]["model_location"]
        else:
            filepath = Path(filepath)
        # merge X_train and y_train
        _X_combined = pd.concat((self._X_train, self._y_train), axis=1)
        _X_combined.to_csv((filepath.parent /
                           (filepath.name)).with_suffix(".csv"), index=False)
        if _config["MODEL"]["save_pickle"] is True:
            with open(filepath.parent / (filepath.name + '.pickle'), "wb") as f:
                pickle.dump(self, f)


def load_model(filepath: str = None) -> Model:
    """Loads a already fitted model from a csv file.    

    Args:
        filepath (str, optional): Location of the model csv file. Defaults to None.

    Returns:
        Model: Fitted Model for creak classification.
    """
    if filepath is None:
        filepath = get_root() / get_config()["MODEL"]["model_location"]
        filepath = (filepath.parent / (filepath.name)).with_suffix(".csv")
    else:
        filepath = Path(filepath)

    if filepath.suffix == ".csv":
        _config = get_config()
        _X_combined = pd.read_csv(filepath)
        model = Model()
        _target_column = _config["MODEL"]["target_label"]
        _feature_columns = _config["MODEL"]["FEATURES"]["for_classification"]
        _X_train, _y_train = _X_combined[_feature_columns], _X_combined[_target_column]
        model.fit(_X_train, _y_train)
        return model
    if filepath.suffix == ".pickle":
        with open(filepath, "rb") as f:
            return pickle.load(f)


clfs = {
    "rfc": RandomForestClassifier,
    "mlp": MLPClassifier
}
