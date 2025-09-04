"""
Functions to compute speaker similarity using pretrained ecapa model implemented in SpeechBrain
"""

import torch
import argparse

import torchaudio
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb").cuda()
classifier.device = "cuda"
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

def calculate_speaker_similarity(wav_fpath_1, wav_fpath_2):
    """
    Compute speaker similarity between two wav files
    """
    global classifier, similarity

    # load wavs
    waveform1, sample_rate1 = torchaudio.load(wav_fpath_1)
    waveform2, sample_rate2 = torchaudio.load(wav_fpath_2)
    assert sample_rate1 == sample_rate2, "Sample rate mismatch"

    # compute embeddings
    embedding1 = classifier.encode_batch(waveform1)
    embedding2 = classifier.encode_batch(waveform2)

    # compute similarity
    out = similarity(embedding1, embedding2)
    return out.item()

def get_speaker_embeddings(wav_fpath):
    """
    Get speaker embeddings from a wav file
    """
    global classifier

    waveform, sample_rate = torchaudio.load(wav_fpath)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
    try:
        embedding = classifier.encode_batch(waveform)
        return embedding
    except:
        return None

def compare_embeddings_within(embs):
    """
    Compare all pairs within a set of speaker embeddings
    and report mean and standard deviation.

    Example:
        embs = torch.cat([emb1, emb2, emb3], dim=0)
        mean, std = compare_embeddings_within(embs)
    """

    embs = embs.cuda()
    # compute speaker similarity
    global similarity
    cos_matrix = similarity(embs, embs.transpose(0,1))

    # initialize a square matrix with zeros on bottom triangle and 1 on top triangle
    mask = torch.triu(torch.ones_like(cos_matrix), diagonal=1)
    cos_flatten = cos_matrix[mask == 1]
    mean = torch.mean(cos_flatten).item()
    std = torch.std(cos_flatten).item()
    return mean, std

def compare_embeddings_across(embs_a, embs_b):
    """
    Compare all pairs between two sets of speaker embeddings
    """
    embs_a = embs_a.cuda()
    embs_b = embs_b.cuda()
    
    cos_matrix = similarity(embs_a, embs_b.transpose(0,1))
    mean = torch.mean(cos_matrix.flatten()).item()
    std = torch.std(cos_matrix.flatten()).item()
    return mean, std

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-1", type=str)
    parser.add_argument("--wav-2", type=str)
    args = parser.parse_args()

    similarity = calculate_speaker_similarity(args.wav_1, args.wav_2)
    print(f"Speaker similarity: {similarity}")