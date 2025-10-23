# SpecUFEx

![SCOPED](https://img.shields.io/endpoint?url=https://runkit.io/wangyinz/scoped/branches/master/Specufex)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Build](https://github.com/ngroebner/specufex/actions/workflows/python-app.yml/badge.svg?branch=main)

SpecUFEx stands for "Unsupervised Spectral Feature Extraction", an unsupervised machine learning algorithm to characterize time variations in spectral content of waveform data. We apply the method to earthquake seismograms. SpecUFEx combines probabilistic non-negative matrix factorization (NMF) and hidden Markov modeling (HMM) of spectrograms (short time Fourier transforms of waveform data) to generate "fingerprints", low dimensional representations of spectral variation through time. Both the NMF and HMM models are fit using stochastic variational inference; the method is therefore scalable to tens or hundreds of thousands of spectrograms. The resulting fingerprints can be used as features for either unsupervised (e.g. clustering) or supervised (e.g. classification) machine learning. The method is described in

[Holtzman, B.K., Paté, A., Paisley, J., Waldhauser, F., Repetto, D.: Machine learning reveals cyclic changes in seismic source spectra in geysers geothermal field. Science advances 4(5) (2018)](https://advances.sciencemag.org/content/4/5/eaao2929)

Please cite this article if you use the package for research purposes.

This repository is a python port of Matlab code originally written by John Paisley at Columbia University. All python code included is written by Ben Holtzman, Theresa Sawi and Nate Groebner.

## Installation

Clone this repository to your computer, cd to the directory, and use `pip` to install.

``` shell
git clone https://github.com/specufex/specufex.git
cd specufex
pip install .
```
If you intend to use the example files with the tutorials, you may have to install [git LFS](https://git-lfs.com) (Large File System) and pull the files. Note: you need to be in te base directory of the repository for the ```git lfs pull``` command to work.

``` shell
git lfs install
git lfs pull
```

Alternatively, a Dockerfile is included that builds a container running Jupyterlab with an environment setup for SpecUFEx. A prebuilt container is available through the [SCOPED](https://github.com/SeisSCOPED) project [here](https://github.com/SeisSCOPED/specufex/pkgs/container/specufex). Or you can directly pull the container if you have Docker:

```bash
docker pull ghcr.io/seisscoped/specufex:latest
```

## Usage

### Fitting models and transforming data

SpecUFEx fits a group of $D x M$ spectrograms, where D is the number of rows (frequency bands) and M is the number of columns (timesteps) in each spectrogram. The spectrograms must be in a numpy-compatible matrix of dimension $N x D x M$, N being the number of spectrograms in the dataset. Each spectrogram must consist of all nonnegative (>=0) entries. (Note, this is not yet checked for.)

The two main classes in this package are `BayesianNonparametricNMF` and `BayesianHMM`. Each has fit, transform, and fit_transform methods to be consistent with the Scikit-learn API style.

The first step is to calculate the nonnegative matrix factorization of your data. This is done by creating a new `BayesianNonParametricNMF` object and calling its `fit` method. This function estimates the model parameters based on all of the data in X.  In the future, we hope to create a convergence criterion based on the ELBO. We iteratively fit the model, one spectrogram at a time, selecting a random spectrogram from our data set. In the example below, `X` is the numpy matrix of spectrograms. Please note that your data must be nonnegative; i.e., all elements must be >= 0.

```python
nmf = BayesianNonparametricNMF(X.shape)

batches = 10000
batch_size = 1

for i in range(batches):
    idx = np.random.randint(X.shape[0], size-batch_size)
    nmf.fit(X[idx])
```

This finds the left matrix of the NMF of the data. Transform the data to the reduced representation, Hs, (the right matrix) via

```python
Vs = nmf.transform(X)
```

Pro tip: a step can be saved by the convenience method `fit_transform, which does the fitting and transformation in one command.  Note, however, that this can take a long time, so you may want to do this in pieces so you can save the resulting NMF left matrix in case something goes wrong (like a power outage).

Next, fit the HMM model with the BayesianHMM class. Currently, in order to setup the object correctly the number of NMF patterns (`num_pat`) and the gain calculated by `BayesianNonparametricNMF` are passed to the constructor.

```python
hmm = BayesianHMM(nmf.num_pat, nmf.gain)

batches = 10000
batch_size = 1

for i in range(batches):
    idx = np.random.randint(Vs.shape[0], size-batch_size)
    hmm.fit(Vs[idx])
```

Similar to the NMF calculation, the data are transformed to fingerprints with the `transform` function.

```python
fingerprints, As, gams = hmm.transform(Vs)
```

Or, if you want to save a step, use `fit_transform` like above.

```python
fingerprints, As, gams = hmm.fit_transform(Vs, nmf.EW)
```

The variable `fingerprints` has the calculated fingerprints (the ultimate matrices of interest), `As` has the state transition matrices of each spectrogram, and `gams` are the state sequence probability matrices.

### Saving and loading models

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

### Development

If you are interested in contributing to SpecUFEx development, please fork this repository and create a new branch for your new code. Create a developemnt environment with conda or virtualenv and install the dev dependencies with `pip install -r requirements-dev.txt`.  Code formatting is done with [ruff](https://docs.astral.sh/ruff/) and is performed on every commit with [pre-commit](https://pre-commit.com). Please write tests for the code you develop. We use [pytest](https://docs.pytest.org/en/7.1.x/) and [nox](https://nox.thea.codes/en/stable/) for writing and running tests. When your code is done, submit a pull request.
