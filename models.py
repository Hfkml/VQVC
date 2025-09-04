import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
#from . import commons
#from . import modules

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x


class Encoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
#create a linear layer for the creak encoder
class CreakEncoder(torch.nn.Module):
    def __init__(self, input_size, model_embedding_size=1024):
        super(CreakEncoder, self).__init__()
        
        self.input_size = input_size
        self.model_embedding_size = model_embedding_size
        
        # Define the linear layer
        self.linear = nn.Linear(input_size, model_embedding_size)
        # Initialize with zeros
        self.linear.weight.data.fill_(0)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        # Perform linear transformation
                # Transpose input tensor to have shape (batch_size, length, 1)
        x = x.permute(0, 2, 1)
        
        # Perform linear transformation
        x = self.linear(x)
        
        # Transpose output tensor to have shape (batch_size, model_embedding_size, length)
        x = x.permute(0, 2, 1)
        
        return x
        
  
        
        
class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=80, model_num_layers=3, model_hidden_size=256, model_embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        
    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames-partial_frames, partial_hop):
            mel_range = torch.arange(i, i+partial_frames)
            mel_slices.append(mel_range)
            
        return mel_slices
    
    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:,-partial_frames:]
        
        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(mel_len, partial_frames, partial_hop)
            mels = list(mel[:,s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)
        
            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            #embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)
        
        return embed


class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    gin_channels,
    ssl_dim,
    use_spk,
    creak,
    cpps,
    h1h2,
    pitch,
    h1a3,
    pitch_var,
    **kwargs):

    super().__init__()
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.gin_channels = gin_channels
    self.ssl_dim = ssl_dim
    self.use_spk = use_spk
    self.creak = creak
    self.cpps = cpps
    self.h1h2 = h1h2
    self.pitch = pitch
    self.h1a3 = h1a3
    self.pitch_var = pitch_var

    self.enc_p = Encoder(ssl_dim, inter_channels, hidden_channels, 5, 1, 16).to(device)
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels).to(device)
    self.enc_q = Encoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels).to(device)
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels).to(device)
    
    if not self.use_spk:
      self.enc_spk = SpeakerEncoder(model_hidden_size=gin_channels, model_embedding_size=gin_channels)
    if self.creak:
        self.enc_creak = CreakEncoder(input_size=1, model_embedding_size=1024).to(device)
        self.enc_creak_flow = CreakEncoder(input_size=1, model_embedding_size=192).to(device)
    if self.cpps:
        self.enc_cpp = CreakEncoder(input_size=1, model_embedding_size=1024).to(device)
        self.enc_cpp_flow = CreakEncoder(input_size=1, model_embedding_size=192).to(device)
    if self.h1h2:
        self.enc_h1h2 = CreakEncoder(input_size=1, model_embedding_size=1024).to(device)
        self.enc_h1h2_flow = CreakEncoder(input_size=1, model_embedding_size=192).to(device)
    if self.pitch:
        self.enc_pitch = CreakEncoder(input_size=1, model_embedding_size=1024).to(device)
        self.enc_pitch_flow = CreakEncoder(input_size=1, model_embedding_size=192).to(device)
    if self.h1a3:
        self.enc_h1a3 = CreakEncoder(input_size=1, model_embedding_size=1024).to(device)
        self.enc_h1a3_flow = CreakEncoder(input_size=1, model_embedding_size=192).to(device)
    if self.pitch_var:
        self.enc_pitch_var = CreakEncoder(input_size=1, model_embedding_size=1024).to(device)
        self.enc_pitch_var_flow = CreakEncoder(input_size=1, model_embedding_size=192).to(device)


  def forward(self, c, spec, g=None, mel=None, c_lengths=None, spec_lengths=None, creaks=None, cpps=None, h1h2s=None, pitches=None, h1a3s=None, pitch_vars=None):
    if c_lengths == None:
      c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
    if spec_lengths == None:
      spec_lengths = (torch.ones(spec.size(0)) * spec.size(-1)).to(spec.device)
    
    if self.creak and creaks is not None:
      c = c + self.enc_creak(creaks)
    if self.cpps and cpps is not None:
      c = c + self.enc_cpp(cpps)
    if self.h1h2 and h1h2s is not None:
        c = c + self.enc_h1h2(h1h2s)
    if self.pitch and pitches is not None:
        c = c + self.enc_pitch(pitches)
    if self.h1a3 and h1a3s is not None:
        c = c + self.enc_h1a3(h1a3s)
    if self.pitch_var and pitch_vars is not None:
        c = c + self.enc_pitch_var(pitch_vars)
      
    if not self.use_spk:
      g = self.enc_spk(mel.transpose(1,2))
    g = g.unsqueeze(-1)
    
    _, m_p, logs_p, _ = self.enc_p(c, c_lengths)

    z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
#    if self.creak and creaks is not None: 
#        z = z + self.enc_creak(creaks)[1]
    z_p = self.flow(z, spec_mask, g=g)
    if self.creak and creaks is not None:
        z_p = z_p + self.enc_creak_flow(creaks)
    if self.cpps and cpps is not None:
        z_p = z_p + self.enc_cpp_flow(cpps)
    if self.h1h2 and h1h2s is not None:
        z_p = z_p + self.enc_h1h2_flow(h1h2s)
    if self.pitch and pitches is not None:
        z_p = z_p + self.enc_pitch_flow(pitches)
    if self.h1a3 and h1a3s is not None:
        z_p = z_p + self.enc_h1a3_flow(h1a3s)
    if self.pitch_var and pitch_vars is not None:
        z_p = z_p + self.enc_pitch_var_flow(pitch_vars)
    z_slice, ids_slice = commons.rand_slice_segments(z, spec_lengths, self.segment_size)
    o = self.dec(z_slice, g=g)
    
    return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

  def infer(self, c, g=None, mel=None, c_lengths=None, creaks=None, cpps=None, h1h2s=None, pitches=None, h1a3s=None, pitch_vars=None):
    if c_lengths == None:
      c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
    if not self.use_spk:
      g = self.enc_spk.embed_utterance(mel.transpose(1,2))
    if self.creak and creaks is not None:
        #c = c + self.enc_creak(creaks)
        c = c + self.enc_creak(creaks.to(c.device))[0]
    if self.cpps and cpps is not None:
        c = c + self.enc_cpp(cpps.to(c.device))[0]
    if self.h1h2 and h1h2s is not None:
        c = c + self.enc_h1h2(h1h2s.to(c.device))[0]
    if self.pitch and pitches is not None:
        c = c + self.enc_pitch(pitches.to(c.device))[0]
    if self.h1a3 and h1a3s is not None:
        c = c + self.enc_h1a3(h1a3s.to(c.device))[0]
    if self.pitch_var and pitch_vars is not None:
       c = c + self.enc_pitch_var(pitch_vars.to(c.device))[0]
    g = g.unsqueeze(-1)

    z_p, m_p, logs_p, c_mask = self.enc_p(c, c_lengths)
    #if self.creak and creaks is not None:
    #    z_p = z_p + self.enc_creak(creaks)[1]
    z = self.flow(z_p, c_mask, g=g, reverse=True)
    if self.creak and creaks is not None:
        z = z + self.enc_creak_flow(creaks)
    if self.cpps and cpps is not None:
        z = z + self.enc_cpp_flow(cpps)
    if self.h1h2 and h1h2s is not None:
        z = z + self.enc_h1h2_flow(h1h2s)
    if self.pitch and pitches is not None:
        z = z + self.enc_pitch_flow(pitches)
    if self.h1a3 and h1a3s is not None:
        z = z + self.enc_h1a3_flow(h1a3s)
    if self.pitch_var and pitch_vars is not None:
        z = z + self.enc_pitch_var_flow(pitch_vars)
    o = self.dec(z * c_mask, g=g)
    
    return o
