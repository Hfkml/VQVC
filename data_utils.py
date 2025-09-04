import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import commons 
from mel_processing import spectrogram_torch, spec_to_mel_torch
from utils import load_wav_to_torch, load_filepaths_and_text, transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#import h5py


"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths, hparams):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length  = hparams.data.filter_length
        self.hop_length     = hparams.data.hop_length
        self.win_length     = hparams.data.win_length
        self.sampling_rate  = hparams.data.sampling_rate
        self.use_sr = hparams.train.use_sr
        self.use_spk = hparams.model.use_spk
        self.spec_len = hparams.train.max_speclen
        self.creak = hparams.model.creak
        self.cpps = hparams.model.cpps
        self.h1h2 = hparams.model.h1h2
        self.pitch = hparams.model.pitch
        self.h1a3 = hparams.model.h1a3
        self.pitch_var = hparams.model.pitch_var

        random.seed(1234)
        random.shuffle(self.audiopaths)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        lengths = []
        for audiopath in self.audiopaths:
            lengths.append(os.path.getsize(audiopath[0]) // (2 * self.hop_length))
        self.lengths = lengths

    def get_creak(self, filename):
        #if not 'no_creak' in filename:
        #    #get a tensor of 1 for each sample of the wavfile
        #    creak = torch.ones(1, os.path.getsize(filename) // (2 * self.hop_length))
        #else:
        #    creak = torch.zeros(1, os.path.getsize(filename) // (2 * self.hop_length))
        #return creak
        
        filename_creak = filename.replace("wavs", "creak")
        creak = np.load(filename_creak.replace(".wav", ".npy"))

        try:
            creak = torch.from_numpy(creak).unsqueeze(0)
        except:
            print(filename)
            raise
        return creak
    def get_cpps(self, filename):
        filename_cpps = filename.replace("wavs", "cpps")
        cpps = np.load(filename_cpps.replace(".wav", ".npy"))
        try:
            cpps = torch.from_numpy(cpps).unsqueeze(0)
        except:
            print(filename)
            raise
        return cpps
    def get_h1h2(self, filename):
        filename_h1h2 = filename.replace("wavs", "norm_db")
        h1h2 = np.load(filename_h1h2.replace(".wav", ".npy"))
        try:
            h1h2 = torch.from_numpy(h1h2).unsqueeze(0)
        except:
            print(filename)
            raise
        return h1h2
    def get_pitch(self, filename):
        filename_pitch = filename.replace("wavs", "pitch")
        pitch = np.load(filename_pitch.replace(".wav", ".npy"))
        try:
            pitch = torch.from_numpy(pitch).unsqueeze(0)
        except:
            print(filename)
            raise
        return pitch

    def get_h1a3(self, filename):
        filename_h1a3 = filename.replace("wavs", "norm_h1a3")
        h1a3 = np.load(filename_h1a3.replace(".wav", ".npy"))
        try:
            h1a3 = torch.from_numpy(h1a3).unsqueeze(0)
        except:
            print(filename)
            raise
        return h1a3
    def get_pitch_var(self, filename):
        filename_pitch_var = filename.replace("wavs", "pitch_var")
        pitch_var = np.load(filename_pitch_var.replace('.wav', '.npy'))
        try:
            pitch_var = torch.from_numpy(pitch_var).unsqueeze(0)
        except:
            print(filename)
            raise
        return pitch_var

    def get_parameters(self, c, spec, audio_norm, spk=None, creak=None, cpps=None, h1h2=None, pitch=None, h1a3=None, pitch_var=None):
        result = [c, spec, audio_norm]  # Start with required parameters

    # Append optional parameters in the specified order
        if self.use_spk:
            result.append(spk)
        if self.creak:
            result.append(creak)
        if self.cpps:
            result.append(cpps)
        if self.h1h2:
            result.append(h1h2)
        if self.pitch:
            result.append(pitch)
        if self.h1a3:
            result.append(h1a3)
        if self.pitch_var:
            result.append(pitch_var)

        return tuple(result)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        if self.creak:
            creak = self.get_creak(filename)
        
        if self.cpps:
            cpps = self.get_cpps(filename)

        if self.h1h2:
            h1h2 = self.get_h1h2(filename)

        if self.pitch:
            pitch = self.get_pitch(filename)
        
        if self.h1a3:
            h1a3 = self.get_h1a3(filename)
            
        if self.pitch_var:
            pitch_var = self.get_pitch_var(filename)
            
        if self.use_spk:
            spk_filename = filename.replace(".wav", ".npy")
            spk_filename = spk_filename.replace("wavs", "spk")
            spk = torch.from_numpy(np.load(spk_filename)).to("cuda")
        
        if not self.use_sr:
            c_filename = filename.replace(".wav", ".pt")
            c_filename = c_filename.replace("wavs", "ssl")
            c = torch.load(c_filename).squeeze(0)
            #current shape is (x, y), but it should be (1024, y) so pad
            if c.size(0) < 1024:
                c = torch.cat([c, torch.zeros(1024 - c.size(0), c.size(1))], 0).to("cuda")
        else:
            i = random.randint(68,92)
            '''
            basename = os.path.basename(filename)[:-4]
            spkname = basename[:4]
            #print(basename, spkname)
            with h5py.File(f"dataset/rs/wavlm/{spkname}/{i}.hdf5","r") as f:
                c = torch.from_numpy(f[basename][()]).squeeze(0)
            #print(c)
            '''
            c_filename = filename.replace(".wav", f"_{i}.pt")
            c_filename = c_filename.replace("DUMMY", "dataset/sr/wavlm")
            c = torch.load(c_filename).squeeze(0)
            
        # 2023.01.10 update: code below can deteriorate model performance
        # I added these code during cleaning up, thinking that it can offer better performance than my provided checkpoints, but actually it does the opposite.
        # What an act of 'adding legs to a snake'!
        '''
        lmin = min(c.size(-1), spec.size(-1))
        spec, c = spec[:, :lmin], c[:, :lmin]
        audio_norm = audio_norm[:, :lmin*self.hop_length]
        _spec, _c, _audio_norm = spec, c, audio_norm
        while spec.size(-1) < self.spec_len:
            spec = torch.cat((spec, _spec), -1)
            c = torch.cat((c, _c), -1)
            audio_norm = torch.cat((audio_norm, _audio_norm), -1)
        start = random.randint(0, spec.size(-1) - self.spec_len)
        end = start + self.spec_len
        spec = spec[:, start:end]
        c = c[:, start:end]
        audio_norm = audio_norm[:, start*self.hop_length:end*self.hop_length]
        '''
        return self.get_parameters(c, spec, audio_norm, spk, creak, cpps, h1h2, pitch, h1a3, pitch_var)

    def __getitem__(self, index):
        return self.get_audio(self.audiopaths[index][0])

    def __len__(self):
        return len(self.audiopaths)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, hps):
        self.hps = hps
        self.use_sr = hps.train.use_sr
        self.use_spk = hps.model.use_spk
        self.creak = hps.model.creak
        self.cpps = hps.model.cpps
        self.h1h2 = hps.model.h1h2
        self.pitch = hps.model.pitch
        self.h1a3 = hps.model.h1a3
        self.pitch_var = hps.model.pitch_var

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]),
            dim=0, descending=True)

        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])
        max_creak_len = max([x[4].size(1) for x in batch])
        max_cpps_len = max([x[5].size(1) for x in batch])
        max_h1h2_len = max([x[6].size(1) for x in batch])
        max_pitch_len = max([x[7].size(1) for x in batch])
        max_h1a3_len = max([x[8].size(1) for x in batch])
        max_pitch_var_len = max([x[9].size(1) for x in batch])

        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        if self.use_spk:
            spks = torch.FloatTensor(len(batch), batch[0][3].size(0))
        if self.creak:
            creak = torch.FloatTensor(len(batch), batch[0][4].size(1))
        if self.cpps:
            cpps = torch.FloatTensor(len(batch), batch[0][5].size(1))
        if self.h1h2:
            h1h2 = torch.FloatTensor(len(batch), batch[0][6].size(1))
        if self.pitch:
            pitch = torch.FloatTensor(len(batch), batch[0][7].size(1))
        if self.h1a3:
            h1a3 = torch.FloatTensor(len(batch), batch[0][8].size(1))
        if self.pitch_var:
            pitch_var = torch.FloatTensor(len(batch), batch[0][9].size(1))
        else:
            spks = None
        
        c_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        creaks_padded = torch.FloatTensor(len(batch), 1, max_creak_len)
        cpps_padded = torch.FloatTensor(len(batch), 1, max_cpps_len)
        #cpps_padded = torch.FloatTensor(len(batch), 1, max_cpp_len)
        h1h2_padded = torch.FloatTensor(len(batch), 1, max_h1h2_len)
        pitch_padded = torch.FloatTensor(len(batch), 1, max_pitch_len)
        h1a3_padded = torch.FloatTensor(len(batch), 1, max_h1a3_len)
        pitch_var_padded = torch.FloatTensor(len(batch), 1, max_pitch_var_len)

        c_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        creaks_padded.zero_()
        cpps_padded.zero_()
        h1h2_padded.zero_()
        pitch_padded.zero_()
        h1a3_padded.zero_()
        pitch_var_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            
            c = row[0]
            c_padded[i, :, :c.size(1)] = c

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
            
            if self.use_spk:
                spks[i] = row[3]
                idx = 4  # Start at index 4 if `use_spk` is True
                if self.creak:
                    creak = row[idx]
                    creaks_padded[i, :, :creak.size(1)] = creak
                    idx += 1  # Move to the next index if `creak` is present
                if self.cpps:
                    cpps = row[idx]
                    cpps_padded[i, :, :cpps.size(1)] = cpps
                    idx += 1  # Move to the next index if `cpps` is present
                if self.h1h2:
                    h1h2 = row[idx]
                    h1h2_padded[i, :, :h1h2.size(1)] = h1h2
                    idx += 1  
                if self.pitch:
                    pitch = row[idx]
                    pitch_padded[i, :, :pitch.size(1)] = pitch
                    idx += 1
                if self.h1a3:
                    h1a3 = row[idx]
                    h1a3_padded[i, :, :h1a3.size(1)] = h1a3
                if self.pitch_var:
                    pitch_var = row[idx]
                    pitch_var_padded[i, :, :pitch_var.size(1)] = pitch_var
                    

            else:
                idx = 3  # Start at index 3 if `use_spk` is False
                if self.creak:
                    creak = row[idx]
                    creaks_padded[i, :, :creak.size(1)] = creak
                    idx += 1  # Move to the next index if `creak` is present
                if self.cpps:
                    cpps = row[idx]
                    cpps_padded[i, :, :cpps.size(1)] = cpps
                    idx += 1  # Move to the next index if `cpps` is present
                if self.h1h2:
                    h1h2 = row[idx]
                    h1h2_padded[i, :, :h1h2.size(1)] = h1h2
                    idx += 1  # Move to the next index if `h1h2` is present
                if self.pitch:
                    pitch = row[idx]
                    pitch_padded[i, :, :pitch.size(1)] = pitch
                    idx += 1
                if self.h1a3:
                    h1a3 = row[idx]
                    h1a3_padded[i, :, :h1a3.size(1)] = h1a3
                if self.pitch_var:
                    pitch_var = row[idx]
                    pitch_var_padded[i, :, :pitch_var.size(1)] = pitch_var



        
        spec_seglen = spec_lengths[-1] if spec_lengths[-1] < self.hps.train.max_speclen + 1 else self.hps.train.max_speclen + 1
        wav_seglen = spec_seglen * self.hps.data.hop_length 

        spec_padded, ids_slice = commons.rand_spec_segments(spec_padded, spec_lengths, spec_seglen)
        wav_padded = commons.slice_segments(wav_padded, ids_slice * self.hps.data.hop_length, wav_seglen)
        
        c_padded = commons.slice_segments(c_padded, ids_slice, spec_seglen)[:,:,:-1]
    
        spec_padded = spec_padded[:,:,:-1]
        wav_padded = wav_padded[:,:,:-self.hps.data.hop_length]

        # Slicing logic
        if self.use_spk and self.creak:
            creaks_padded = commons.slice_segments(creaks_padded, ids_slice, spec_seglen)[:, :, :-1]
            if self.cpps:
                cpps_padded = commons.slice_segments(cpps_padded, ids_slice, spec_seglen)[:, :, :-1]
            if self.h1h2:
                h1h2_padded = commons.slice_segments(h1h2_padded, ids_slice, spec_seglen)[:, :, :-1]
            if self.pitch:
                pitch_padded = commons.slice_segments(pitch_padded, ids_slice, spec_seglen)[:, :, :-1]
            if self.h1a3:
                h1a3_padded = commons.slice_segments(h1a3_padded, ids_slice, spec_seglen)[:, :, :-1]
            if self.pitch_var:
                pitch_var_padded = commons.slice_segments(pitch_var_padded, ids_slice, spec_seglen)[:, :, :-1]

        elif self.use_spk:
            if self.cpps:
                cpps_padded = commons.slice_segments(cpps_padded, ids_slice, spec_seglen)[:, :, :-1]
            if self.h1h2:
                h1h2_padded = commons.slice_segments(h1h2_padded, ids_slice, spec_seglen)[:, :, :-1]
            if self.pitch:
                pitch_padded = commons.slice_segments(pitch_padded, ids_slice, spec_seglen)[:, :, :-1]
            if self.h1a3:
                h1a3_padded = commons.slice_segments(h1a3_padded, ids_slice, spec_seglen)[:, :, :-1]
            if self.pitch_var:
                pitch_var_padded = commons.slice_segments(pitch_var_padded, ids_slice, spec_seglen)[:, :, :-1]

        # Return logic
        output = [c_padded.to(device), spec_padded.to(device), wav_padded.to(device)]

        if self.use_spk:
            output.append(spks.to(device))
        if self.creak:
            output.append(creaks_padded.to(device))
        if self.cpps:
            output.append(cpps_padded.to(device))
        if self.h1h2:
            output.append(h1h2_padded.to(device))
        if self.pitch:
            output.append(pitch_padded.to(device))
        if self.h1a3:
            output.append(h1a3_padded.to(device))
        if self.pitch_var:
            output.append(pitch_var_padded.to(device))

        return tuple(output)



class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
