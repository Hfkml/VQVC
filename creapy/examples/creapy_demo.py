# %% [markdown]
# # creapy demo
# 
# This is a simple demonstration notebook to show the classification process using creapy
# 
# First, define the audio- and respective textgrid path

# %%
import creapy
from pathlib import Path
# %%
#example_folder_path = Path(__file__).parent
audio_path = "../../data/out/creaky_vctk.wav"
textgrid_path = "../../data/out/creaky_vctk_whisper/p232_180.TextGrid"

# %% [markdown]
# Normally, `creapy` will use the model that is trained of both genders. However, you may change this and choose the model trained on `male` / `female` speakers only by setting the `gender` variable to `male` or `female` respectively or keep it unchanged (`None`).

# %%
X_test, y_pred, sr = creapy.process_file(audio_path, textgrid_path=textgrid_path, gender_model='female')

# %% [markdown]
# To change parameters you can either change them manually in the config file (see ReadME) or with the `set_config` function. In the ReadMe is a list of the configuratable parameters.

# %%
creapy.reset_config()
#creapy.set_config(gender_model = 'female', tier_name ='creapy', zcr_threshold = 0.09)

# %% [markdown]
# Creapy can also process more than one file at a time. This can be done with the `process_folder` function. This function will not return the computed features and the creak probability in contrast to `process_file`.

# %%
#creapy.process_folder(example_folder_path / "../audio", example_folder_path / "../textgrids")

# %% [markdown]
# The plot function renders the features from `X_test` and the creak probability `y_pred` over time. One can use the scroll bar to search for the desired time and look at the computed features. Most of the time, only the features `creak_probability`, `zcr` and `ste` are interesting for the classification process (to toggle features, just click on the label on the right).

# %%
creapy.plot(X_test, y_pred, sr)


