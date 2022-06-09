import numpy as np
from scipy.stats import gamma
from scipy.io import savemat, loadmat
#from testclasses import BayesianNonparametricNMF, BayesianHMM
import os
from specufex import *
import argparse

def test_nmf_fit():
    np.random.seed(0)

    # just load 100 spectrograms from NW geysers dataset
    X = np.load('./tests/NW_spectrograms.npy')[:100]

    nmf = BayesianNonparametricNMF(X.shape, num_pat=40)
    nmf.fit(X)
    Vs, Xpwrs = nmf.transform(X)


def python_v_octave():
    # run specufex all the way through with a deterministic W1 matrix in NMF,
    # deterministic B1 in HMM, and same order of spectrograms for fitting
    # Do this for the python and Matlab implementations and compare results
    X = np.load('../../kilauea_spectrograms.npy')[:2]
    print(X.shape)
    savemat('data.mat',{'X':X})

    dim = X.shape[1]
    num_pat = 75
    N_eff = 1

    np.random.seed(10)

    W1 = (N_eff/1000)*100*gamma.rvs(10*np.ones((dim,num_pat)),1/10)

    savemat('W1.mat', {'W1':W1})

    # B1 will have to be created when NMF is done in order to know num_pat
    # B1 = B1 = (N/n)*1000*gamma.rvs(np.ones((self.num_state, num_pat)),1)

    # Below is custom functions

    nmf = BayesianNonparametricNMF()
    nmf.fit(X, W1, NbSteps=1, verbose=1)
    savemat('pythonEW.mat', {'EW':nmf.EW})

    Hs,Xpwrs = nmf.transform(X[:2])
    savemat('pythonHsXpwrs.mat', {'Hs':Hs, 'Xpwrs':Xpwrs})

    num_pat = nmf.EW.shape[1]
    num_state = 15

    B1 = 1000*gamma.rvs(np.ones((num_state, num_pat)),1)

    savemat('B1.mat', {'B1':B1})

    hmm = BayesianHMM(num_state=num_state, verbose=1)
    hmm.fit(Hs[:2], nmf.EW, B1)
    savemat('pythonEB.mat', {'EB':hmm.EB})

    fprints, As, Ppis = hmm.transform(Hs[:2])
    savemat('pythonPrints.mat', {'As':As,'Ppis':Ppis,'fprints':fprints})

    # run octave script here
    print('Running octave tests')
    os.system('octave octavetest.m')

    octaveprints = loadmat('octavePrints.mat')

    print(np.allclose(np.moveaxis(octaveprints['A2s'], -1, 0),fprints))


if __name__ == '__main__':
    test_nmf_fit()