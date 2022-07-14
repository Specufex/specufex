.. toctree::
    :caption: Contents

SpecUFEx
========

SpecUFEx stands for "Spectral Unsupervised Feature Extraction."
It is an algorithm for feature extraction from seismic spectrograms
(and other nonnegative matrices) developed by John Paisley and first presented in
`Holtzman, et. al, 2018 <https://advances.sciencemag.org/content/4/5/eaao2929>`_.
The SpeUFEx algorithm consists of two models:

1. A probabilistic nonparametric nonnegative matrix factorization of a group of spectrograms (short time Fourier transforms of waveform data),
2. A probabilistic hidden Markov model of the reduced matrices.

Both of these models are fit using stochastic variational inference [Hoff2013]_,
and the method is therefore scalable to tens or hundreds of thousands
of spectrograms. Once the HMM model is fit, the state transition
probabilities for the hidden state sequence of each reduced spectrogram
are calculated. These transition probabilities then constitute the
features of each spectrogram. The fingerprints can then be used as input to a variety of unsupervised and supervised machine learning methods.

.. [Hoff2013] Ref to Hoffman, Blei, Wang, Paisley 2013