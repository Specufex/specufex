.. toctree::
    :caption: Contents

Usage
======

Fitting models and transforming data
-------------------------------------

SpecUFEx fits a group of N $D x M$ spectrograms, D is the number of rows (frequency bands) and M is columns (timesteps). The spectrograms must be in a numpy-compatible matrix of dimension $N x D x M$, N being the number of spectrograms in the dataset. The matrix must consist of all nonnegative (>=0) entries. (Note, this is not yet checked for.)

The two main classes in this package are `BayesianNonparametricNMF` and `BayesianHMM`. Each has fit, transform, and fit_transform methods to be consistent with the Scikit-learn API style.

The first step is to preprocess your data. For this example, we use the function used in Holtzman et. al, which is included in `utilities.py` of the package. X is our dataset.::

    from SpecUFEx import BayesianNonparametricNMF, BayesianHMM, normalize_spectrogram

    Xis = []
    for Xi in X:
        Xi = normalize_spectrogram(Xi)
        Xis.append(Xi)
    X = np.stack(Xis, 0)

Next, find the nonnegative matrix factorization of the normalized data.
This is simply done by creating a new `BayesianNonParametricNMF` object and
calling its `fit` method. This function estimates the model parameters
based on all of the data in X. Batch learning can be done by splitting
your data matrix into minibatches. In the future, we hope to create a
convergence criterion based on the ELBO.::

    nmf = BayesianNonparametricNMF()
    nmf.fit(X)

This finds the left matrix of the NMF of the data. Transform the data
to the reduced representation, Hs, (the right matrix) via::

    Vs, Xpwrs = nmf.transform(X)

Pro tip: a step can be saved by the convenience method `fit_transform`,
which does the fitting and transformation in one command.  Note, however,
that this can take a long time, so you may want to do this in pieces
so you can save the resulting NMF left matrix in case something goes wrong (
like a power outage).

Next, fit the HMM is with the BayesianHMM class. Currently, in order to
setup the object correctly the number of NMF patterns (`num_pat`) and
the gain calculated by `BayesianNonparametricNMF` are passed to the constructor.::

    hmm = BayesianHMM(nmf.num_pat, nmf.gain)
    hmm.fit(Vs)

Similar to the NMF calculation, the data is transformed to fingerprints
with the `transform` function.::

    fingerprints, As, Ppis = hmm.transform(Vs)

Or, if you want to save a step, use `fit_transform` like above.:

    fingerprints, As, Ppis = hmm.fit_transform(Vs)

The variable `fingerprints` has the calculated fingerprints (the ultimate
matrices of interest), `As` has the state transition matrices of each
spectrogram, and `Ppis` has the initial state vectors.

Saving and loading models
-------------------------

Once you have fit either the NMF or HMM model (or both!) you can
save the parameters for the model using built in functions. From
the above examples where `nmf` and `hmm` are objects that contain
trained models, simply use::

    nmf.save(filename)
    hmm.save(filename)

to save the parameters. Likewise, to load an already saved model
and instantiate a new model object use::

    nmf = BayesianNonparametricNMF.load(filename)
    hmm = BayesianHMM.load(filename)

and now you have NMF and HMM models that are ready to transform your data.
