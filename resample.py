import librosa
import numpy as np
import os
import argparse
import soundfile as sf

#resample
def resample(input_path, output_path, sr):
    y, sr = librosa.load(input_path, sr=sr)
    target_sr = 16000
    y = librosa.resample(y, sr, target_sr)
    sf.write(output_path, y, target_sr)

#main

#create argparser
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='resample wav files to 16kHz')
    parser.add_argument('--input_dir', type=str, help='input directory')
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--sr', type=int, help='source sample rate')

    args = parser.parse_args()

    resample(args.input_dir, args.output_dir, args.sr)

