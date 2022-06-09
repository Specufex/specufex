# python version of SVI NMF

import numpy as np
from scipy.stats import gamma
from scipy.special import psi
import h5py
from .modelutils import SaveableModel
import numexpr as ne


class BayesianNonparametricNMF(SaveableModel):
    """
    Python implementation of Stochastic variational
    inference non negative matrix factorization.
    Adapted from original Matlab implementation by
    John Paisley. Expects normalized matrices.

    Parameters
    -----------
    input_dims: tuple (int, int, int)
        Dimensions of the spectrograms to fit.
        (number of spectrograms x frequency bands x timesteps)
    num_pat: int
        Initial number of NMF patterns
    """

    def __init__(self, input_dims, num_pat = 75):

        assert(len(input_dims)==3), "input_dims must be length 3"
        self.input_dims = input_dims
        self.N_eff, self.dim, _ = input_dims
        self.num_pat = num_pat

        # setup initial matrices
        self.h01 = 1/self.num_pat
        self.h02 = 1

        self.w01 = 10/self.dim
        self.w02 = 1

        self.a01 = 1/self.num_pat
        self.a02 = 1

        W1 = gamma.rvs(10*np.ones((self.dim,self.num_pat)),1/10)
        self.W1 = (self.N_eff/1000)*100*W1
        self.W2 = (self.N_eff/1000)*100*(np.ones((self.dim,self.num_pat)))
        self.EW = self.W1 / self.W2

        self.A1 = (self.N_eff/1000)*10*np.ones((1,self.num_pat))
        self.A2 = (self.N_eff/1000)*10*np.ones((1,self.num_pat))
        self.step = -1 # to make the steps work out

    def fit(self, X, y=None, resort_args=True, stepsize=None, verbose=0):
        """Fits the SVI NMF model to data.
        Can fit incrementally on batches.

        Arguments
        ----------
        X: 3 dimensional numpy array
            The array of spectrograms to fit. Randomize order
            before passing to this function.
        y: 3 dimensional numpy array, optional
            Optional sample of data for calculation of loss. **Not implemented yet.**
        resort_args: bool, default=True
            Whether to resort the NMF matrix by frequency. Increases interpretability
            of the result.
        stepsize: float, optional
            Constant step size (learning rate) for SVI. If not specified,
            a predetermined annealing schedule will be used.
        verbose: int, default=0
            0 for no output
            1 to output current step and number of patterns
            2 or higher to output step, patterns, annealing temp, and step size

        Returns
        ----------
        None
            Saves the EW, EA, EWlnA, and gain arrays to the BayesianNonparametricNMF object.
        """

        for Xi in X:
            # annealing temp -  track total steps so annealing is correct
            T = 1 + 0.75**(self.step -1)

            X_reshape = Xi.T[np.newaxis,...]
            n_tsteps = X_reshape.shape[1]

            # Perhaps save some calculations
            #  calc A1./A2, would only save one matrix multiplication
            #  calc N_eff/n_tsteps would save 3 floating point calcs

            V1 = np.ones((self.num_pat, n_tsteps))/self.num_pat
            V2 = (self.h02 + np.sum(self.EW*(self.A1/self.A2),0)[:,np.newaxis])/T

            ElnWA = psi(self.W1) - np.log(self.W2) + psi(self.A1)-np.log(self.A2)
            ElnWA_reshape = ElnWA.T[:,np.newaxis,:]

            t1 = np.max(ElnWA_reshape, 0)
            ElnWA_reshape = ElnWA_reshape - t1

            # update P  P(V1), V1(P)  ; T is the annealing variable-- it decreases to 1 with iterations.
            for t in range(int(10+np.round(1+5*(T-1))-1)):
                ElnV = (psi(V1) - int(t > 0)*np.log(V2))
                P = ElnWA_reshape/T + ElnV[...,np.newaxis]/T
                P = ne.evaluate("exp(P)")
                P = P / np.sum(P,0)
                V1 = 1 + (self.h01 + np.sum(P*X_reshape,2) - 1)/T

            ElnV = psi(V1) - np.log(V2)
            P = ElnWA_reshape/T + ElnV[...,np.newaxis]/T
            P = ne.evaluate("exp(P)")
            P = P / np.sum(P,0)

            if not stepsize:
                rho = (250/(1+5*(T-1)) + self.step)**(-.51)
            else:
                rho = stepsize
            # N is the number of spectrograms n_tsteps is per step number
            # P is a hidden variable purely for optimization
            W1_up = self.w01 + (self.N_eff/n_tsteps) * np.sum(X_reshape*P, 1).T
            W2_up = ((V1/V2)*(self.A1/self.A2).T).sum(1)
            W2_up = self.w02 + (self.N_eff/n_tsteps) * W2_up.T

            # W1_up and W2_up are improved Ws based on the subset of spectrograms used in that step... (SVI)
            W1 = (1-rho)*self.W1 + rho*(1+(W1_up-1)/T)  # weighted average of the new and the past
            W2 = (1-rho)*self.W2 + rho*W2_up/T

            A1_up = self.a01 + (self.N_eff/n_tsteps) * (X_reshape*P).sum(2).sum(1).T
            A2_up = self.a02 + (self.N_eff/n_tsteps) * (W1/W2).sum(0)*(V1/V2).sum(1).T

            A1 = (1-rho)*self.A1 + rho*(1+(A1_up-1)/T)
            A2 = (1-rho)*self.A2 + rho*A2_up/T

            # throw away useless patterns
            idx_prune = ((A1/A2) > 0.01)[0]
            self.W1 = W1[:,idx_prune]
            self.W2 = W2[:,idx_prune]
            self.A1 = A1[:,idx_prune]
            self.A2 = A2[:,idx_prune]
            self.EW = self.W1/self.W2
            self.num_pat = self.W1.shape[1]

            self.step += 1
            if verbose == 1:
                print(f"step {self.step+1}/{len(X)}, num patterns: {self.A1.shape[1]}")
            elif verbose > 1:
                print(f"step {self.step+1}/{len(X)}, num patterns: {self.A1.shape[1]},  T: {T}, rho: {rho}")
        # Sort EW and EA by frequency for more interpretable result
        if resort_args:
            sorted_args = np.argsort(np.argmax(self.EW, axis=0))
            self.EW = self.EW[:,sorted_args]
            self.W1 = self.W1[:,sorted_args]
            self.W2 = self. W2[:,sorted_args]

            self.A1 = self.A1[:,sorted_args]
            self.A2 = self.A2[:,sorted_args]

        self.EA = self.A1 / self.A2
        self.ElnWA = psi(self.W1) - np.log(self.W2) + psi(self.A1)-np.log(self.A2)
        self.gain = self.EW.sum(0)
        #self.num_pat = self.EW.shape[1]

        # for when we have the subsample loss figured out
        #loss = loss_fnc(y)
        #return loss

    def transform(self, X):
        """Calculate individual V matrices based on the NMF model.

        Arguments
        ----------
        X: 3 dimensional numpy array
            Array of X matrices to trransform into V (activation) Matrices.

        Returns
        ----------
        Numpy array
            The activation matrices (V's) for each spectrogram in X.
        """
        n_tsteps_old = 0
        Vs = []
        Xpwrs = []

        for Xi in X:
            # This is guaranteed to run at least once (and define tempMat)
            # since n_tsteps will be > 0
            n_tsteps = Xi.shape[1]

            # this should only be necessary if spectrograms have different lengths
            # should get rid of this
            if n_tsteps != n_tsteps_old:
                ElnWA_reshape = self.ElnWA.T[:, np.newaxis, :]
                t1 = np.max(ElnWA_reshape,axis=0)
                ElnWA_reshape = ElnWA_reshape - t1[:,np.newaxis,:]
                tempMat = self.h02 + np.sum(self.EW*(self.A1/self.A2),0)[:,np.newaxis]
                n_tsteps_old = n_tsteps

            V1 = np.ones((self.num_pat,n_tsteps))
            Xi_reshape = Xi.T[np.newaxis,...]
            V2 = tempMat
            for t in range(5):
                ElnV = psi(V1) - np.log(V2)
                P = ElnWA_reshape + ElnV[...,np.newaxis]
                P = ne.evaluate("exp(P)")
                P = P / P.sum(0)
                V1 = self.h01 + np.sum(P*Xi_reshape,2)

            V = V1/V2
            Xpwr = Xi.sum(0)
            Vs.append(V)
            Xpwrs.append(Xpwr)

        Vs = np.moveaxis(np.dstack(Vs), -1, 0)
        #self.Xpwrs = np.squeeze(Xpwrs)

        return Vs

    def fit_transform(self, X, y=None, verbose=0):
        """Fits the SVI NMF model to data, then transforms.
        Not for use with minibatch fitting.

        Arguments
        ----------
        X: 3 dimensional numpy array
            The array of spectrograms to fit. Randomize order
            before passing to this function.
        y: 3 dimensional numpy array, optional
            Optional sample of data for calculation of loss. **Not implemented yet.**
        resort_args: bool, default=True
            Whether to resort the NMF matrix by frequency. Increases interpretability
            of the result.
        stepsize: float, optional
            Constant step size (learning rate) for SVI. If not specified,
            a predetermined annealing schedule will be used.
        verbose: int
            0 for no output
            1 to output current step and number of patterns
            2 or higher to output step, patterns, annealing temp, and step size

        Returns
        ----------
        Numpy array
            The activation matrices (V's) for each spectrogram in X.

        """

        self.fit(X, y=None, verbose=verbose)
        if verbose > 0:
            print('NMF fit, now calculating features')
        return self.transform(X)

    @classmethod
    def load(cls, filename):
        """Loads a BayesianNonparametricNMF model from an hdf5 file.

        Arguments
        ----------
        filename: string
            Name (and path) of file containing the parameters for the model.

        Returns
        ----------
        BayesianNonparametricNMF
            BayesianNonparametricNMF object initialized with parameters and ready to transform data.

        """
        with h5py.File(filename, 'r') as hf:
            # init the class with the input_dims stored in the parameter file
            return cls(hf['input_dims'][()])._load(filename)
