import os
from pathlib import Path

import h5py
import numpy as np

from specufex import BayesianHMM, BayesianNonparametricNMF


class TestNMF:
    def setup(self):

        # for the NMF tests
        N, D, M = 10, 5, 7
        self.X = np.random.poisson(lam=1.0, size=(N, D, M))
        self.nmf = BayesianNonparametricNMF(self.X.shape)
        if os.path.exists("tests/tested_save_nmf.h5"):
            os.remove("tests/tested_save_nmf.h5")

    def teardown(self):
        if os.path.exists("tests/tested_save_nmf.h5"):
            os.remove("tests/tested_save_nmf.h5")

    def test_fit(self):
        """Test the NMF fit method. Checks that the calculated EW and EA
        matrices when multiplied have same first dimension as X.
        """

        # default verbosity
        self.nmf.fit(self.X)
        fit_shape = self.nmf.EW @ np.diag(self.nmf.EA[0])
        assert self.X.shape[1] == fit_shape.shape[0]

        # verbosity 1
        self.nmf.fit(self.X, verbose=1)
        fit_shape = self.nmf.EW @ np.diag(self.nmf.EA[0])
        assert self.X.shape[1] == fit_shape.shape[0]

        # verbosity > 1
        self.nmf.fit(self.X, verbose=46)
        fit_shape = self.nmf.EW @ np.diag(self.nmf.EA[0])
        assert self.X.shape[1] == fit_shape.shape[0]

    def test_fit_no_resort_args(self):
        """Test the NMF fit method without resorting the As. Checks that
        the calculated EW and EA matrices when multiplied have same first
        dimension as X.
        """
        self.nmf.fit(self.X, resort_args=False)
        fit_shape = self.nmf.EW @ np.diag(self.nmf.EA[0])

        assert self.X.shape[1] == fit_shape.shape[0]

    def test_fit_stepsize(self):
        """Test the NMF fit method when stepsize is specified.
        Checks that the calculated EW and EA matrices when multiplied
        have same first dimension as X.
        """
        self.nmf.fit(self.X, stepsize=0.01)
        fit_shape = self.nmf.EW @ np.diag(self.nmf.EA[0])

        assert self.X.shape[1] == fit_shape.shape[0]

    def test_transform(self):
        """Test the NMF transform method. Checks that when a transformed
        V matrix is multiplied by EWA it has same dimension as
        X
        """
        self.nmf.fit(self.X)
        Vs = self.nmf.transform(self.X)
        est_X = self.nmf.EW @ np.diag(self.nmf.EA[0]) @ Vs[0]
        assert self.X[0].shape == est_X.shape

    def test_fit_transform(self):
        """Test the NMF transform method. Checks that when a transformed
        V matrix is multiplied by EWA it has same dimension as
        X
        """

        # default verbosity
        Vs = self.nmf.fit_transform(self.X)
        est_X = self.nmf.EW @ np.diag(self.nmf.EA[0]) @ Vs[0]
        assert self.X[0].shape == est_X.shape

        # verbosity > 0
        Vs = self.nmf.fit_transform(self.X, verbose=1)
        est_X = self.nmf.EW @ np.diag(self.nmf.EA[0]) @ Vs[0]
        assert self.X[0].shape == est_X.shape

    def test_save(self):
        """test NMF model save"""
        self.nmf.fit(self.X)
        assert self.nmf.save("tests/tested_save_nmf.h5", overwrite=False)
        os.remove("tests/tested_save_nmf.h5")

    def test_overwrite_save(self):
        """test NMF model save when file is present"""
        self.nmf.fit(self.X)
        Path("tests/tested_save_nmf.h5").touch(exist_ok=False)
        assert not self.nmf.save("tests/tested_save_nmf.h5", overwrite=False)
        os.remove("tests/tested_save_nmf.h5")

    def test_load(self):
        """Test model loading"""
        nmf = BayesianNonparametricNMF.load("tests/test_nmf_params.h5")
        assert isinstance(nmf, BayesianNonparametricNMF)

        with h5py.File("tests/test_nmf_params.h5") as hf:
            assert (nmf.EW == hf["EW"][()]).all()
            assert (nmf.EA == hf["EA"][()]).all()
            # might want to add tests for all the parameters - do the member thing


class TestHMM:
    def setup(self):
        """set up"""
        N, num_pat, timesteps = 10, 5, 7
        self.V = np.random.poisson(lam=1.0, size=(N, num_pat, timesteps))
        gain = np.ones(num_pat)

        self.hmm = BayesianHMM(num_pat, gain)

        if os.path.exists("tests/tested_save_hmm.h5"):
            os.remove("tests/tested_save_hmm.h5")

    def teardown(self):
        if os.path.exists("tests/tested_save_hmm.h5"):
            os.remove("tests/tested_save_hmm.h5")

    def test_fit_default(self):
        """test the HMM fit method"""

        # default verbosity
        self.hmm.fit(self.V)
        assert self.hmm.EB.shape == (self.hmm.num_state, self.hmm.num_pat)

        # verbosity > 0
        self.hmm.fit(self.V, verbose=23)
        assert self.hmm.EB.shape == (self.hmm.num_state, self.hmm.num_pat)

    def test_fit_EB_dist_sort(self):
        """Test that EB is sorted by pairwise distances.
        Currently only checks that the shape is correct as above.
        """
        self.hmm.fit(self.V, resort_EB="distance")

        assert self.hmm.EB.shape == (self.hmm.num_state, self.hmm.num_pat)

    def test_fit_EB_energy_sort(self):
        """Test that EB is sorted by decreasing energy, i.e., gradient is nonpositive"""
        self.hmm.fit(self.V, resort_EB="energy")
        energies = self.hmm.EB.sum(axis=1)
        energies_diff = np.diff(energies)
        assert np.all(energies_diff <= 0)

    def test_getStateMatrices(self):
        """Test the _getStateMatrices method. Only checks to see
        if matrix dimensions are correct.
        """
        self.hmm.fit(self.V, resort_EB="energy")
        As, ppis, gams = self.hmm._getStateMatrices(self.V)
        assert As.shape == (self.V.shape[0], self.hmm.num_state, self.hmm.num_state)
        assert ppis.shape == (self.V.shape[0], self.hmm.num_state)
        assert gams.shape == (self.V.shape[0], self.hmm.num_state, self.V.shape[2])

    def test_getFingerprints(self):
        """Test the _getFingerprints method"""
        self.hmm.fit(self.V, resort_EB="energy")
        As, ppis, _ = self.hmm._getStateMatrices(self.V)
        fprints = self.hmm._getFingerprints(As, ppis)
        assert fprints.shape == (
            self.V.shape[0],
            self.hmm.num_state,
            self.hmm.num_state,
        )

    def test_transform(self):
        """test the HMM transform method"""
        self.hmm.fit(self.V, resort_EB="energy")
        fprints, As, gams = self.hmm.transform(self.V)
        assert fprints.shape == (
            self.V.shape[0],
            self.hmm.num_state,
            self.hmm.num_state,
        )
        assert As.shape == (self.V.shape[0], self.hmm.num_state, self.hmm.num_state)
        assert gams.shape == (self.V.shape[0], self.hmm.num_state, self.V.shape[2])

    def test_fit_transform(self):
        """Test fit_transform method"""
        fprints, As, gams = self.hmm.fit_transform(self.V)
        assert fprints.shape == (
            self.V.shape[0],
            self.hmm.num_state,
            self.hmm.num_state,
        )
        assert As.shape == (self.V.shape[0], self.hmm.num_state, self.hmm.num_state)
        assert gams.shape == (self.V.shape[0], self.hmm.num_state, self.V.shape[2])

    def test_save(self):
        """test HMM model save"""
        self.hmm.fit(self.V)
        assert self.hmm.save("tests/tested_save_hmm.h5", overwrite=False)
        os.remove("tests/tested_save_hmm.h5")
        # assert os.path.exists("tests/tested_save_hmm.h5")

    def test_overwrite_save(self):
        """test HMM model save when file is present"""
        self.hmm.fit(self.V)
        Path("tests/tested_save_hmm.h5").touch(exist_ok=False)
        assert not self.hmm.save("tests/tested_save_hmm.h5", overwrite=False)
        os.remove("tests/tested_save_hmm.h5")

    def test_load(self):
        """test HMM model loading"""
        hmm = BayesianHMM.load("tests/test_hmm_params.h5")
        assert isinstance(hmm, BayesianHMM)

        with h5py.File("tests/test_hmm_params.h5") as hf:
            assert (hmm.EB == hf["EB"][()]).all()
            # self.assertTrue( (hmm.EA==hf['EA'][()]).all() )
            # might want to add tests for all the parameters - do the member thing
