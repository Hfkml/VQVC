from pathlib import Path
import sys
import logging
import os
import time
import librosa
import numpy as np
import torch
from scipy.io.wavfile import write
from tqdm import tqdm
import utils
from models import SynthesizerTrn
from speaker_encoder.voice_encoder import SpeakerEncoder
from wavlm import WavLM, WavLMConfig
from datetime import datetime

def generate_timestamp():
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string suitable for a filename
    #timestamp = now.strftime("%Y%m%d_%H%M%S")
    timestamp = now.strftime("%Y%m%d_%H%M%S_") + str(int(now.microsecond / 1000)).zfill(2)
    # Return the formatted timestamp with the desired file extension
    return timestamp

if len(sys.argv) != 7:
    print("Usage: script.py <session_number> <system_persona affective/not_affective> <tts_system tacotron/tacotron_prosody/matcha>")
    sys.exit(1)
source_path = sys.argv[1]
target_path1 = sys.argv[2]
target_path2 = sys.argv[3]
scaling_factor = float(sys.argv[4])
output_audio = sys.argv[5]
creak_value = int(sys.argv[6])

wavlm_large_path = 'wavlm/WavLM-Large.pt'
freevc_chpt_path = 'logs/freevc2/G_190000.pth'

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

def get_cmodel():
    checkpoint = torch.load(wavlm_large_path)
    cfg = WavLMConfig(checkpoint['cfg'])
    cmodel = WavLM(cfg)
    cmodel.load_state_dict(checkpoint['model'])
    cmodel.eval()

    return cmodel

hps = utils.get_hparams_from_file('configs/freevc.json')
os.makedirs('/'.join(sys.argv[5].split('/')[:-1]), exist_ok=True)

net_g = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
)

utils.load_checkpoint(freevc_chpt_path, net_g, optimizer=None, strict=True)
cmodel = get_cmodel()
smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt', device='cpu')

#print("converting...")
wav_tgt1, _ = librosa.load(target_path1, sr=hps.data.sampling_rate)
wav_tgt1, _ = librosa.effects.trim(wav_tgt1, top_db=20)
wav_tgt2, _ = librosa.load(target_path2, sr=hps.data.sampling_rate)
wav_tgt2, _ = librosa.effects.trim(wav_tgt2, top_db=20)
g_tgt1 = smodel.embed_utterance(wav_tgt1)
g_tgt2 = smodel.embed_utterance(wav_tgt2)
g_tgt = scaling_factor*g_tgt1 + (1-scaling_factor)*g_tgt2
g_tgt = torch.from_numpy(g_tgt).unsqueeze(0)

# src
wav_src, _ = librosa.load(source_path, sr=hps.data.sampling_rate)
wav_src = torch.from_numpy(wav_src).unsqueeze(0)

c = utils.get_content(cmodel, wav_src)

tgt_audio = net_g.infer(c, g=g_tgt, creaks=torch.tensor(np.zeros((1, 1, (wav_src.size(1)//320-1))), dtype=torch.float32)+creak_value)
tgt_audio = tgt_audio[0][0].data.cpu().float().numpy()

timestamp = generate_timestamp()
print(timestamp)
write(output_audio, hps.data.sampling_rate, tgt_audio)