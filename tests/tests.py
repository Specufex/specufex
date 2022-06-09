from specufex import BayesianNonparametricNMF, BayesianHMM
import numpy as np
import os
import pytest
import h5py


class TestNMF:

    def setup(self):

        # for the NMF tests
        N,D,M=10,5,7
        self.X = np.random.poisson(lam=1.0, size=(N,D,M))
        self.nmf = BayesianNonparametricNMF(self.X.shape)
        if os.path.exists("tested_save_nmf.h5"):
            os.remove("tested_save_nmf.h5")

    def teardown(self):
        if os.path.exists("tested_save_nmf.h5"):
            os.remove("tested_save_nmf.h5")

    def test_fit(self):
        """Test the NMF fit method. Checks that the calculated EW and EA
        matrices when multiplied have same first dimension as X.
        """
        self.nmf.fit(self.X)
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

    def test_save(self):
        """Test the NMF save method"""
        self.nmf.fit(self.X)
        self.nmf.save('tested_save_nmf.h5', overwrite=False)

        assert os.path.exists('tested_save_nmf.h5')

    def test_load(self):
        """Test model loading"""
        nmf = BayesianNonparametricNMF.load('test_nmf_params.h5')
        assert isinstance(nmf, BayesianNonparametricNMF)

        with h5py.File('test_nmf_params.h5') as hf:
            assert (nmf.EW==hf['EW'][()]).all()
            assert (nmf.EA==hf['EA'][()]).all()
            # might want to add tests for all the parameters - do the member thing


class TestHMM:

    def setup(self):
        """set up"""
        N,num_pat,timesteps=10,5,7
        self.V = np.random.poisson(lam=1.0, size=(N,num_pat,timesteps))
        gain = np.ones(num_pat)

        self.hmm = BayesianHMM(num_pat, gain)

        if os.path.exists("tested_save_hmm.h5"):
            os.remove("tested_save_hmm.h5")

    def teardown(self):
        if os.path.exists("tested_save_hmm.h5"):
            os.remove("tested_save_hmm.h5")

    def test_fit_default(self):
        """test the HMM fit method"""
        self.hmm.fit(self.V)

        assert self.hmm.EB.shape == (self.hmm.num_state, self.hmm.num_pat)

    def test_fit_EB_dist_sort(self):
        """test the HMM fit method"""
        self.hmm.fit(self.V, resort_EB="distance")

        assert self.hmm.EB.shape == (self.hmm.num_state, self.hmm.num_pat)

    def test_fit_EB_energy_sort(self):
        """test the HMM fit method"""
        self.hmm.fit(self.V, resort_EB="energy")

        assert self.hmm.EB.shape == (self.hmm.num_state, self.hmm.num_pat)

    def test_getStateMatrices(self):
        pass

    def test_getFingerprints(self):
        pass

    def test_transform(self):
        """test the HMM transform method"""
        pass

    def test_save(self):
        """test HMM model method"""
        self.hmm.fit(self.V)
        self.hmm.save('tested_save_hmm.h5', overwrite=False)

        assert os.path.exists('tested_save_hmm.h5')

    def test_load(self):
        """test HMM model loading"""
        hmm = BayesianHMM.load('test_hmm_params.h5')
        assert isinstance(hmm, BayesianHMM)

        with h5py.File('test_hmm_params.h5') as hf:
            assert (hmm.EB==hf['EB'][()]).all() 
            #self.assertTrue( (hmm.EA==hf['EA'][()]).all() )
            # might want to add tests for all the parameters - do the member thing