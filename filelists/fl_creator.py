import glob
from pathlib import Path
import os

fls = os.listdir('../libri_train/games/wavs')

with open('train_games.txt', 'w') as f:
    for fl in fls:
        f.write(f"./libri_train/games/wavs/{fl}\n")