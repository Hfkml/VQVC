from __future__ import annotations

import creapy

if __name__ == "__main__":
    print("Extracting features..")
    _config = creapy.get_config()
    X, y = creapy.calculate_features_for_folder("/home/creaker/tip/audio/samples_new",
                                 features=_config["MODEL"]["FEATURES"]["for_classification"])
    
    print("Fitting model..")
    model = creapy.Model()
    model.fit(X, y)
    model.save()
