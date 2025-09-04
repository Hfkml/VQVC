from glob import glob
from pathlib import Path
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch
import os
import torch



wavs = glob('/nfs/deepspeech/home/lameris/libri_train/train/wavs/*.wav')
specs = glob('/nfs/deepspeech/home/lameris/libri_train/train/wavs/*.spec.pt')

wavs_stem = [Path(wav).stem for wav in wavs]
specs_stem = [Path(spec).stem for spec in specs]

for i, wav in enumerate(wavs_stem):
    if wav not in specs_stem:
        print(wav)
        audio, sampling_rate = load_wav_to_torch(wavs[i])
        if sampling_rate != 16000:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, 16000))
        audio_norm = audio / 32768.0
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = wavs[i].replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            print(f"Spec already exists: {spec_filename}")
        else:
            spec = spectrogram_torch(audio_norm, 1280,
                sampling_rate, 320, 1280,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)