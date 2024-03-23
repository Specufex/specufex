import os
import pytest
import numpy as np

from specufex import BayesianHMM, BayesianNonparametricNMF

@pytest.fixture
def setup_nmf():
    # for the NMF tests
    N, D, M = 10, 5, 7
    X = np.random.poisson(lam=1.0, size=(N, D, M))
    nmf = BayesianNonparametricNMF(X.shape)
    if os.path.exists("tests/tested_save_nmf.h5"):
        os.remove("tests/tested_save_nmf.h5")
    return [X, nmf]

def teardown():
    if os.path.exists("tests/tested_save_nmf.h5"):
        os.remove("tests/tested_save_nmf.h5")

def test_nmf_fit(setup_nmf):
    """Test the NMF fit method. Checks that the calculated EW and EA
    matrices when multiplied have same first dimension as X.
    """

    X, nmf = setup_nmf
    # default verbosity
    nmf.fit(X)
    fit_shape = nmf.EW @ np.diag(nmf.EA[0])
    assert X.shape[1] == fit_shape.shape[0]

    # verbosity 1
    nmf.fit(X, verbose=1)
    fit_shape = nmf.EW @ np.diag(nmf.EA[0])
    assert X.shape[1] == fit_shape.shape[0]

    # verbosity > 1
    nmf.fit(X, verbose=46)
    fit_shape = nmf.EW @ np.diag(nmf.EA[0])
    assert X.shape[1] == fit_shape.shape[0]

def test_fit_no_resort_args(setup_nmf):
    """Test the NMF fit method without resorting the As. Checks that
    the calculated EW and EA matrices when multiplied have same first
    dimension as X.
    """
    X, nmf = setup_nmf

    nmf.fit(X, resort_args=False)
    fit_shape = nmf.EW @ np.diag(nmf.EA[0])

    assert X.shape[1] == fit_shape.shape[0]

def test_fit_stepsize(setup_nmf):
    """Test the NMF fit method when stepsize is specified.
    Checks that the calculated EW and EA matrices when multiplied
    have same first dimension as X.
    """
    X, nmf = setup_nmf

    nmf.fit(X, stepsize=0.01)
    fit_shape = nmf.EW @ np.diag(nmf.EA[0])

    assert X.shape[1] == fit_shape.shape[0]

def test_transform(setup_nmf):
    """Test the NMF transform method. Checks that when a transformed
    V matrix is multiplied by EWA it has same dimension as
    X
    """
    X, nmf = setup_nmf
    nmf.fit(X)
    Vs = nmf.transform(X)
    est_X = nmf.EW @ np.diag(nmf.EA[0]) @ Vs[0]
    assert X[0].shape == est_X.shape

def test_fit_transform(setup_nmf):
    """Test the NMF transform method. Checks that when a transformed
    V matrix is multiplied by EWA it has same dimension as
    X
    """
    X, nmf = setup_nmf
    # default verbosity
    Vs = nmf.fit_transform(X)
    est_X = nmf.EW @ np.diag(nmf.EA[0]) @ Vs[0]
    assert X[0].shape == est_X.shape

    # verbosity > 0
    Vs = nmf.fit_transform(X, verbose=1)
    est_X = nmf.EW @ np.diag(nmf.EA[0]) @ Vs[0]
    assert X[0].shape == est_X.shape

@pytest.fixture
def setup_hmm():
    """set up"""
    N, num_pat, timesteps = 10, 5, 7
    V = np.random.poisson(lam=1.0, size=(N, num_pat, timesteps))
    gain = np.ones(num_pat)

    hmm = BayesianHMM(num_pat, gain)

    if os.path.exists("tests/tested_save_hmm.h5"):
        os.remove("tests/tested_save_hmm.h5")

    return [V, hmm]

def teardown(self):
    if os.path.exists("tests/tested_save_hmm.h5"):
        os.remove("tests/tested_save_hmm.h5")


def test_hmm_fit_default(setup_hmm):
    """test the HMM fit method"""

    V, hmm = setup_hmm
    # default verbosity
    hmm.fit(V)
    assert hmm.EB.shape == (hmm.num_state, hmm.num_pat)

    # verbosity > 0
    hmm.fit(V, verbose=23)
    assert hmm.EB.shape == (hmm.num_state, hmm.num_pat)

def test_fit_EB_dist_sort(setup_hmm):
    """Test that EB is sorted by pairwise distances.
    Currently only checks that the shape is correct as above.
    """
    V, hmm = setup_hmm
    hmm.fit(V, resort_EB="distance")

    assert hmm.EB.shape == (hmm.num_state, hmm.num_pat)

def test_fit_EB_energy_sort(setup_hmm):
    """Test that EB is sorted by decreasing energy, i.e., gradient is nonpositive"""
    V, hmm = setup_hmm
    hmm.fit(V, resort_EB="energy")
    energies = hmm.EB.sum(axis=1)
    energies_diff = np.diff(energies)
    assert np.all(energies_diff <= 0)

def test_getStateMatrices(setup_hmm):
    """Test the _getStateMatrices method. Only checks to see
    if matrix dimensions are correct.
    """
    V, hmm = setup_hmm
    hmm.fit(V, resort_EB="energy")
    As, ppis, gams = hmm._getStateMatrices(V)
    assert As.shape == (V.shape[0], hmm.num_state, hmm.num_state)
    assert ppis.shape == (V.shape[0], hmm.num_state)
    assert gams.shape == (V.shape[0], hmm.num_state, V.shape[2])

def test_getFingerprints(setup_hmm):
    """Test the _getFingerprints method"""
    V, hmm = setup_hmm
    hmm.fit(V, resort_EB="energy")
    As, ppis, _ = hmm._getStateMatrices(V)
    fprints = hmm._getFingerprints(As, ppis)
    assert fprints.shape == (
        V.shape[0],
        hmm.num_state,
        hmm.num_state,
    )

def test_transform(setup_hmm):
    """test the HMM transform method"""
    V, hmm = setup_hmm
    hmm.fit(V, resort_EB="energy")
    fprints, As, gams = hmm.transform(V)
    assert fprints.shape == (
        V.shape[0],
        hmm.num_state,
        hmm.num_state,
    )
    assert As.shape == (V.shape[0], hmm.num_state, hmm.num_state)
    assert gams.shape == (V.shape[0], hmm.num_state, V.shape[2])

def test_fit_transform(setup_hmm):
    """Test fit_transform method"""
    V, hmm = setup_hmm
    fprints, As, gams = hmm.fit_transform(V)
    assert fprints.shape == (
        V.shape[0],
        hmm.num_state,
        hmm.num_state,
    )
    assert As.shape == (V.shape[0], hmm.num_state, hmm.num_state)
    assert gams.shape == (V.shape[0], hmm.num_state, V.shape[2])