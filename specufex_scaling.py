# get time scaling for specufex
# run tutorial data included with specufex
# scaling with number of waveforms, spectrogram size for model fitting
# scaling with number of waveforms, spectrogram size for transform
# do 100,000 fit iterations for NMF and HMM
# do for 2 matrix sizes
# 

import datetime
from time import time

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import scipy.signal as sp
from tqdm import trange

from specufex import BayesianNonparametricNMF, BayesianHMM

# waveform parameters
fs = 500 #Hz
len_data = 10000

# bandpass filter
fMin = 5
fMax = 150

# spectrogram parameters
sgramMode='magnitude'
sgramScaling='spectrum'

# frequency/time resolution
nperseg = np.array([256, 128, 64])
noverlap = nperseg/4
nfft = 1024



if __name__ == "__main__":
   cat = pd.read_pickle("waveforms.pkl").iloc[:1000]
   print(cat.head())
   print(f"{len(cat.trace)} waveforms in file")

   # result structure
   results = {
       "spect_size": [],
       "spectrogram_time": [],
       "nmf_batches": [],
       "nmf_batch_size": [],
       "nmf_fit_time": [],
       "nmf_transform_time": [],
       "hmm_batches": [],
       "hmm_batch_size": [],
       "hmm_fit_time": [],
       "hmm_transform_time": [],
       "total_time": [] 
   }
   t_init = time()
   t0 = t_init

   # spectrogram and n waveforms scaling:

   for i in range(3):
       fSTFT, tSTFT, STFT_raw = sp.spectrogram(
           x=np.stack(cat["trace"].values),
           fs=fs,
           nperseg=nperseg[i],
           noverlap=noverlap[i],
           nfft=nfft,
           scaling=sgramScaling,
           axis=-1,
           mode=sgramMode
       )
       print(STFT_raw.shape)
       freq_slice = np.where((fSTFT >= fMin) & (fSTFT <= fMax))
       fSTFT   = fSTFT[freq_slice]
       STFT_0 = STFT_raw[:,freq_slice,:].squeeze()
       normConstant = np.median(STFT_0, axis=(1,2))
       STFT_norm = STFT_0 / normConstant[:,np.newaxis,np.newaxis]  # norm by median
       del STFT_0
       STFT_dB = 20*np.log10(STFT_norm, where=STFT_norm != 0) # convert to dB
       del STFT_norm
       STFT = np.maximum(0, STFT_dB) # make sure nonnegative
       del STFT_dB
       cat["stft"] = list(STFT)
       bad_idx = cat["stft"][cat["stft"].apply(lambda x: np.isnan(x).any())].index
       print(f"Bad spectrograms: \n{cat.loc[bad_idx].name}")
       cat = cat.drop(bad_idx).sort_values("name")
       print("spect time:", time() - t0)
       results["spectrogram_time"].append(time() - t0)
       results["spect_size"].append(STFT_raw.shape)
       t0 = time()

       # NMF fit
       batches = 1000
       batch_size = 1
       results["nmf_batches"].append(batches)
       results["nmf_batch_size"].append(batch_size)
       nmf = BayesianNonparametricNMF(np.stack(cat["stft"].values).shape, num_pat=40)
       t = trange(batches, desc="NMF fit progress ", leave=True)
       for i in t:
           idx = np.random.randint(len(cat["stft"].values), size=batch_size)
           nmf.fit(cat["stft"].iloc[idx].values)
           t.set_postfix_str(f"Patterns: {nmf.num_pat}")
       t1 = time()
       results["nmf_fit_time"].append(t1 - t0)
       t0 = t1

       # NMF transform
       Vs = nmf.transform(cat["stft"].values)
       t1 = time()
       results["nmf_transform_time"].append(t1 - t0)
       t0 = t1

       #HMM fit 
       num_states = 6
       hmm = BayesianHMM(nmf.num_pat, nmf.gain, num_state=num_states, Neff=46000)
       batches = 1000
       batch_size = 1
       results["hmm_batches"].append(batches)
       results["hmm_batch_size"].append(batch_size)
       t = trange(batches, desc="HMM fit progress ", leave=True)
       for i in t:
           idx = np.random.randint(Vs.shape[0], size=1)
           hmm.fit(Vs[idx])
       t1 = time()
       results["hmm_fit_time"].append(t1-t0)
       t0 = t1

       #HMM transform
       fingerprints, As, gams = hmm.transform(Vs)
       t1 = time()
       results["hmm_transform_time"].append(t1-t0)
       results["total_time"].append(t1-t_init)

       print(results)

       pd.DataFrame(results).to_csv("results.csv")