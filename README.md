# SpecUFEx

SpecUFEx stands for "Unsupervised Spectral Feature Extraction", an unsupervised machine learning algorithm to characterize time variations in spectral content of waveform data. We apply the method to earthquake seismograms. SpecUFEx combines probabilistic non-negative matrix factorization (NMF) and hidden Markov modeling (HMM) of spectrograms (short time Fourier transforms of waveform data) to generate "fingerprints", low dimensional representations of spectral variation through time. Both the NMF and HMM models are fit using stochastic variational inference; the method is therefore scalable to tens or hundreds of thousands of spectrograms. The resulting fingerprints can be used as features for either unsupervised (e.g. clustering) or supervised (e.g. classification) machine learning. The method is described in

[Holtzman, B.K., Paté, A., Paisley, J., Waldhauser, F., Repetto, D.: Machine learning reveals cyclic changes in seismic source spectra in geysers geothermal field. Science advances 4(5) (2018)](https://advances.sciencemag.org/content/4/5/eaao2929)

Please cite this article if you use the package for research purposes.

This repository is a python port of Matlab code originally written by John Paisley at Columbia University. All python code included is written by Ben Holtzman, Theresa Sawi and Nate Groebner.

## Installation

Clone this repository to your computer, cd to the directory, and use `pip` to install.

``` shell
git clone https://github.com/nategroebner/specufex.git
cd specufex
pip install .
```

## Usage

### Fitting models and transforming data

SpecUFEx fits a group of $D x M$ spectrograms, where D is the number of rows (frequency bands) and M is the number of columns (timesteps) in each spectrogram. The spectrograms must be in a numpy-compatible matrix of dimension $N x D x M$, N being the number of spectrograms in the dataset. Each spectrogram must consist of all nonnegative (>=0) entries. (Note, this is not yet checked for.)

The two main classes in this package are `BayesianNonparametricNMF` and `BayesianHMM`. Each has fit, transform, and fit_transform methods to be consistent with the Scikit-learn API style.

The first step is to preprocess your data. For this example, we use the function used in Holtzman et. al, which is included in `utilities.py` of the package. X is our dataset.

```shell
from SpecUFEx import BayesianNonparametricNMF, BayesianHMM, normalize_spectrogram

Xis = []
for Xi in X:
    Xi = normalize_spectrogram(Xi)
    Xis.append(Xi)
X = np.stack(Xis, 0)
```

Next, find the nonnegative matrix factorization of the normalized data. This is simply done by creating a new `BayesianNonParametricNMF` object and calling its `fit` method. This function estimates the model parameters based on all of the data in X. Batch learning can be done by splitting your data matrix into minibatches. In the future, we hope to create a convergence criterion based on the ELBO.

```shell
nmf = BayesianNonparametricNMF(X.shape)
nmf.fit(X)
```

This finds the left matrix of the NMF of the data. Transform the data to the reduced representation, Hs, (the right matrix) via

`Vs, Xpwrs = nmf.transform(X)`

Pro tip: a step can be saved by the convenience method `fit_transform, which does the fitting and transformation in one command.  Note, however, that this can take a long time, so you may want to do this in pieces so you can save the resulting NMF left matrix in case something goes wrong (like a power outage).

Next, fit the HMM model with the BayesianHMM class. Currently, in order to setup the object correctly the number of NMF patterns (`num_pat`) and the gain calculated by `BayesianNonparametricNMF` are passed to the constructor.

```shell
hmm = BayesianHMM(nmf.num_pat, nmf.gain)
hmm.fit(Vs)
```

Similar to the NMF calculation, the data are transformed to fingerprints with the `transform` function.

`fingerprints, As, Ppis = hmm.transform(Vs)`

Or, if you want to save a step, use `fit_transform` like above.

`fingerprints, As, Ppis = hmm.fit_transform(Vs, nmf.EW)`

The variable `fingerprints` has the calculated fingerprints (the ultimate matrices of interest), `As` has the state transition matrices of each spectrogram, and `Ppis` has the initial state vectors.

### Saving and loading models

*Please note: The following is a work in progress. If the following code isn't working for you, the trained model classes can be pickled and saved to disk.*

Once you have fit either the NMF or HMM model (or both!) you can save the parameters for the model using built in functions. From the above examples where `nmf` and `hmm` are objects that contain trained models, simply use

```shell
nmf.save(filename)
hmm.save(filename)
```

to save the parameters. Likewise, to load an already saved model and instatiate a new model object use

```shell
nmf = BayesianNonparametricNMF.load(filename)
hmm = BayesianHMM.load(filename)
```

and now you have NMF and HMM models that are ready to transform your data.
