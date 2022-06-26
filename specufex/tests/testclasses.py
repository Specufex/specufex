import numpy as np
from scipy.stats import gamma
from scipy.special import psi
from numpy.linalg import inv, matrix_power
from scipy.linalg import fractional_matrix_power

np.set_printoptions(precision=4)


class BayesianNonparametricNMF:
    """
    Python implementation of Stochastic variational
    inference non negative matrix factorization.
    Adapted from original Matlab implementation by
    John Paisley. Expects normalized matrices.

    init variables here.
    """

    def __init__(self, num_pat=75, N_eff=1):

        self.num_pat = num_pat
        self.N_eff = N_eff

    def fit(self, X, W1, NbSteps=1, verbose=0):
        """Fits the SVI NMF model to data

        arguments:
        X - nspectrograms x nfrequencies X ntimesteps array
        NbSteps - number of timesteps to fit on
        W1 - A matrix of Gamma distributed random variables
             Used to make the algorithm deterministic for debugging
        """

        # dim = rows (frequency bands in spectrogram)
        # N =  number of spectrograms
        self.N, self.dim, _ = X.shape

        # setup initial matrices
        self.h01 = 1 / self.num_pat
        self.h02 = 1

        self.w01 = 10 / self.dim
        self.w02 = 1

        self.a01 = 1 / self.num_pat
        self.a02 = 1

        # W1 = gamma.rvs(10*np.ones((self.dim,self.num_pat)),1/10)
        self.W1 = W1
        self.W2 = (self.N_eff / 1000) * 100 * (np.ones((self.dim, self.num_pat)))

        self.A1 = (self.N_eff / 1000) * 10 * np.ones((1, self.num_pat))
        self.A2 = (self.N_eff / 1000) * 10 * np.ones((1, self.num_pat))

        for step, x in enumerate(X):
            T = 1 + 0.75 ** (step - 1)  # annealing temp
            X_sample = x  # uncomment for production

            # DEBUG uncomment below for production
            # X_reshape = np.hstack(X_sample).T[np.newaxis,:,:]
            X_reshape = X_sample.T[np.newaxis, :, :]
            N_batch = X_reshape.shape[1]

            #######
            # In the original code, the spectrograms are normalized *after* they are stacked
            # Probably so they have the same normnalization
            # Do we need to do stack and normalize them in the dataloader?
            # But there are so few steps where we operate on more than one
            # spectrogram so it hardly seems worth it
            #######
            # Perhaps save some calculations
            #  calc A1./A2, would only save one matrix multiplication
            #  calc N_eff/N_batch would save 3 floating point calcs

            self.EW = self.W1 / self.W2

            H1 = np.ones((self.num_pat, N_batch)) / self.num_pat
            H2 = (
                self.h02 + np.sum(self.EW * (self.A1 / self.A2), 0)[:, np.newaxis]
            ) / T

            ElnWA = psi(self.W1) - np.log(self.W2) + psi(self.A1) - np.log(self.A2)
            ElnWA_reshape = ElnWA.T[:, np.newaxis, :]

            t1 = np.max(ElnWA_reshape, 0)
            ElnWA_reshape = ElnWA_reshape - t1
            # print("ElnWA_reshape shape ", ElnWA_reshape.shape)

            # update P  P(H1), H1(P)  ; T is the annealing variable-- it decreases to 1 with iterations.
            for t in range(int(10 + np.round(1 + 5 * (T - 1)) - 1)):
                ElnH = psi(H1) - int(t > 0) * np.log(H2)
                P = ElnWA_reshape / T + ElnH[..., np.newaxis] / T
                P = np.exp(P)
                P = P / np.sum(P, 0)
                H1 = 1 + (self.h01 + np.sum(P * X_reshape, 2) - 1) / T

            ElnH = psi(H1) - np.log(H2)
            P = ElnWA_reshape / T + ElnH[..., np.newaxis] / T
            P = np.exp(P)
            P = P / np.sum(P, 0)
            # print("P shape", P.shape)

            rho = (250 / (1 + 5 * (T - 1)) + step) ** (-0.51)
            # N is the number of spectrograms N_batch is per step number
            # P is a hidden variable purely for optimization
            W1_up = self.w01 + (self.N_eff / N_batch) * np.sum(X_reshape * P, 1).T
            W2_up = ((H1 / H2) * (self.A1 / self.A2).T).sum(1)
            W2_up = self.w02 + (self.N_eff / N_batch) * W2_up.T

            # W1_up and W2_up are improved Ws based on the subset of spectrograms used in that step... (SVI)
            W1 = (1 - rho) * self.W1 + rho * (
                1 + (W1_up - 1) / T
            )  # weighted average of the new and the past
            W2 = (1 - rho) * self.W2 + rho * W2_up / T

            A1_up = (
                self.a01 + (self.N_eff / N_batch) * (X_reshape * P).sum(2).sum(1).T
            )  # [np.newaxis,...]
            A2_up = (
                self.a02
                + (self.N_eff / N_batch) * (W1 / W2).sum(0) * (H1 / H2).sum(1).T
            )

            A1 = (1 - rho) * self.A1 + rho * (1 + (A1_up - 1) / T)
            A2 = (1 - rho) * self.A2 + rho * A2_up / T

            idx_prune = ((A1 / A2) > 0.001)[0]
            self.W1 = W1[:, idx_prune]
            self.W2 = W2[:, idx_prune]
            self.A1 = A1[:, idx_prune]
            self.A2 = A2[:, idx_prune]
            self.num_pat = self.W1.shape[1]

            if verbose > 0:
                print(
                    "step {}/{}, num patterns: {}".format(
                        step + 1, NbSteps, self.A1.shape[1]
                    )
                )

        self.EW = self.W1 / self.W2
        # Sort EW and EA by frequency for more interpretable result
        resort_args = np.argsort(np.argmax(self.EW, axis=0))
        self.EW = self.EW[:, resort_args]
        self.W1 = self.W1[:, resort_args]

        self.A1 = self.A1[:, resort_args]
        self.A2 = self.A2[:, resort_args]
        self.EA = self.A1 / self.A2

        self.ElnWA = psi(self.W1) - np.log(self.W2) + psi(self.A1) - np.log(self.A2)

        self.gain = self.EW.sum(0)
        self.num_pat = self.EW.shape[1]

    def transform(self, X):
        """Calculate individual H matrices."""
        N_batch_old = 0
        Hs = []
        Xpwrs = []

        for Xi in X:
            # This is guaranteed to run at least once (and define tempMat)
            # since N_batch will be > 0
            N_batch = Xi.shape[1]

            if N_batch != N_batch_old:
                ElnWA_reshape = self.ElnWA.T[:, np.newaxis, :]
                t1 = np.max(ElnWA_reshape, axis=0)
                ElnWA_reshape = ElnWA_reshape - t1[:, np.newaxis, :]
                tempMat = (
                    self.h02 + np.sum(self.EW * (self.A1 / self.A2), 0)[:, np.newaxis]
                )
                N_batch_old = N_batch

            H1 = np.ones((self.num_pat, N_batch))
            Xi_reshape = Xi.T[np.newaxis, ...]
            H2 = tempMat  # moved this out of following loop
            for t in range(5):
                ElnH = psi(H1) - np.log(H2)
                P = ElnWA_reshape + ElnH[..., np.newaxis]
                P = np.exp(P)
                P = P / P.sum(0)
                H1 = self.h01 + np.sum(P * Xi_reshape, 2)

            H = H1 / H2
            Xpwr = Xi.sum(0)
            # save([write_loc_NMF 'out.' files(file).name],'H','Xpwr','gain');
            Hs.append(H)
            Xpwrs.append(Xpwr)

        Hs = np.moveaxis(np.dstack(Hs), -1, 0)
        Xpwrs = np.squeeze(Xpwrs)

        return [Hs, Xpwrs]

    def fit_transform(self, X, NbSteps=1, verbose=0):
        """Fit and transform the given matrices."""

        self.fit(X, NbSteps=NbSteps, verbose=verbose)
        if verbose > 0:
            print("NMF fit, now calculating features")
        return self.transform(X)
        if verbose > 0:
            print("Done")


class BayesianHMM:
    def __init__(self, num_state=15, max_inner_ite=20, verbose=0):
        self.num_state = num_state
        self._max_inner_ite = max_inner_ite
        self.verbose = verbose

    def fit(self, X, EW, B1, NbSteps=1):
        """Fit the hidden Markov model

        Arguments:
            X (numpy array of float): 3-dimensional numpy array containing
                the activation matrices calculated in the NMF step.
                (n_matrices x num_pat x timesteps)
            EW (2D numpy matrix): W matrix from Bayesian NMF step.
                Perhaps we can get rid of this at some point, then
                We could make the HMM model independent of the NMF model???
            NbSteps (int): number of iterations for the fit.
                Note: The algorithm will run for this many steps.
                There is no convergence criterion.

        Returns:
            self
        """

        """Matrix dimensions

        dim:     rows in the EW matrix, also num_pat
        gain:    columns in the EW matrix
        X, H:    the activation matrix, num_pat(dim) x timesteps
        len_seq: timesteps in H matrix (and spectrogram)
        B:       num_state x dim - Gamma distributed matrix
        A:       num_state x num_state - Markov transition matrix,
                 each row independently Dirichlet distributed
        lnPrH:   num_state x timesteps (len_seq)
        Ppi:     num_state - initial-state distribution

        """
        N = 1

        num_pat = EW.shape[1]
        self.gain = EW.sum(0)

        tauPpi = 0.01 / self.num_state
        tauA = 0.1 / self.num_state
        tau1 = 0.1
        tau2 = 0.1

        n = 1

        B2 = (N / n) * 1000 * 0.03 * np.ones((self.num_state, num_pat))

        for step, Xi in enumerate(X):
            print("Current step is {}/{}".format(step, NbSteps))
            n = int(np.round(1 + 10 * 0.75 ** (step - 1)))
            ElnB = psi(B1) - np.log(B2)
            EB = B1 / B2
            Bup1 = np.zeros_like(B1)
            Bup2 = np.zeros_like(B2)
            Atmp = np.zeros((self.num_state, self.num_state))
            for i in range(n):
                # choose random matrix
                # random_index = np.random.randint(0, high=N)
                # random_index = i  #sequential - for debugging,delete for production *****************************
                H = Xi  # individual H matrix
                len_seq = H.shape[1]  # number of timesteps
                # gain is the EW matrix summed along 0th dimension
                H2 = np.diag(self.gain) @ H  # scaling the right matrix, for HMM

                A = np.ones((self.num_state, self.num_state))
                Ppi = np.ones(self.num_state)
                for ite in range(self._max_inner_ite):
                    expElnA = np.exp(
                        psi(A) - psi(A.sum(1))
                    )  # convert each row to prob dist, find exp of ElnA
                    expElnPi = np.exp(
                        psi(Ppi) - psi(Ppi.sum())
                    )  # same for Pi (int dist)
                    lnPrH = ElnB @ H2 - EB.sum(1)[:, np.newaxis]
                    lnPrH = (
                        lnPrH - np.max(lnPrH, axis=0)[np.newaxis, :]
                    )  # convert to prob by dividing by max
                    explnPrH = np.exp(lnPrH)

                    # forward-backward
                    alpha = np.zeros((self.num_state, len_seq))
                    beta = np.zeros((self.num_state, len_seq))
                    alpha[:, 0] = expElnPi * explnPrH[:, 0]
                    beta[:, len_seq - 1] = 1
                    for s in range(1, len_seq):
                        alpha[:, s] = (expElnA.T @ alpha[:, s - 1]) * explnPrH[:, s]
                        alpha[:, s] = alpha[:, s] / np.sum(alpha[:, s])
                        beta[:, len_seq - s - 1] = expElnA @ (
                            beta[:, len_seq - s] * explnPrH[:, len_seq - s]
                        )
                        beta[:, len_seq - s - 1] = beta[:, len_seq - s] / np.sum(
                            beta[:, len_seq - s]
                        )
                    gam = alpha * beta
                    gam = gam / gam.sum(0)
                    Ppi = tauPpi + gam[:, 0]
                    A = 0 * A + tauA
                    for s in range(1, len_seq):
                        mat = expElnA * np.outer(
                            alpha[:, s - 1], beta[:, s] * explnPrH[:, s]
                        )
                        mat = mat / mat.sum()
                        A = A + mat
                        if ite == self._max_inner_ite - 1:
                            # Bup1 = Bup1 + gam[:,s][:,np.newaxis] @ H2[:,s][np.newaxis,:]
                            # Bup1 = Bup1 + gam[:,s] @ H2[:,s].T
                            Bup1 = Bup1 + np.outer(gam[:, s], H2[:, s])
                            Bup2 = Bup2 + gam[:, s][:, np.newaxis]

                Atmp = Atmp + A
                Bup1 = Bup1 + np.outer(gam[:, 0], H2[:, 0])
                Bup2 = Bup2 + gam[:, 0][:, np.newaxis]

            rho = (250 / (1 + 5 * (n - 1)) + step) ** (-0.51)
            self.B1 = (1 - rho) * B1 + rho * (tau1 + (N / n) * Bup1)
            self.B2 = (1 - rho) * B2 + rho * (tau2 + (N / n) * Bup2)
            self.EB = self.B1 / self.B2
            self.ElnB = psi(self.B1) - np.log(self.B2)

    def getStateMatrices(self, X):
        """
        Create the state transition matrices for each activation matrix
        """

        # [num_state,dim] = size(B1);
        tauPpi = 0.01 / self.num_state
        tauA = 0.1 / self.num_state
        tau1 = 0.1
        tau2 = 0.1

        As = []
        Ppis = []

        for Xi in X:
            # print('Current step is {}/{}'.format())

            len_seq = Xi.shape[1]
            H2 = np.diag(self.gain) @ Xi

            A = np.ones((self.num_state, self.num_state))
            Ppi = np.ones(self.num_state)
            for ite in range(self._max_inner_ite):
                expElnA = np.exp(psi(A) - psi(A.sum(1)))
                expElnPi = np.exp(psi(Ppi) - psi(Ppi.sum()))
                lnPrH = self.ElnB @ H2 - self.EB.sum(1)[:, np.newaxis]
                lnPrH = lnPrH - np.max(lnPrH, axis=0)[np.newaxis, :]
                explnPrH = np.exp(lnPrH)

                # forward-backward
                alpha = np.zeros((self.num_state, len_seq))
                beta = np.zeros((self.num_state, len_seq))
                alpha[:, 0] = expElnPi * explnPrH[:, 0]
                beta[:, len_seq - 1] = 1
                for s in range(1, len_seq):
                    alpha[:, s] = (expElnA.T @ alpha[:, s - 1]) * explnPrH[:, s]
                    alpha[:, s] = alpha[:, s] / np.sum(alpha[:, s])
                    beta[:, len_seq - s - 1] = expElnA @ (
                        beta[:, len_seq - s] * explnPrH[:, len_seq - s]
                    )
                    beta[:, len_seq - s - 1] = beta[:, len_seq - s] / np.sum(
                        beta[:, len_seq - s]
                    )

                gam = alpha * beta
                gam = gam / gam.sum(0)
                Ppi = tauPpi + gam[:, 0]
                A = 0 * A + tauA
                for s in range(1, len_seq):
                    # mat = expElnA*(alpha[:,s-1]*(beta[:,s]*explnPrH[:,s]).T)
                    mat = expElnA * np.outer(
                        alpha[:, s - 1], beta[:, s] * explnPrH[:, s]
                    )
                    mat = mat / mat.sum()
                    A = A + mat

            As.append(A)
            Ppis.append(Ppi)

        # make sure the dimension fo the below are (nsamples x num_state (x num_state))
        As = np.stack(As, axis=0)
        Ppis = np.squeeze(Ppis)

        return [As, Ppis]

    def getFingerprints(self, As, Ppis):
        """
        docstring
        """
        eps = np.finfo(float).eps

        fingerprints = []
        # Pinorm and Anorm calcs are vectorized
        T = 100

        # This loop might be able to vectorized - np.linalg.inv boradcasts over batch (1st) dimension
        for i in range(As.shape[0]):
            Ppi_len = len(Ppis[i])
            Ppi = Ppis[i] - 0.01 / Ppi_len

            A = As[i] - (0.1 - eps) / Ppi_len
            Pinorm = Ppi / Ppi.sum()
            Anorm = A / A.sum(1)[:, np.newaxis]
            Eye = np.eye(Ppi_len)
            a = Eye - T / (T + 1) * Anorm
            b = Eye - matrix_power((T / (T + 1) * Anorm), (T + 1))
            statvec = (
                Pinorm.T
                @ inv(np.eye(Ppi_len) - T / (T + 1) * Anorm)
                @ ((np.eye(Ppi_len)) - (T / T + 1) * Anorm) ** (T + 1)
            )
            statvec = statvec.T / statvec.sum()
            # numpy gives inaccurate (at least different from Matalb, which I think is correct)
            # matrix power results for high powers and matrices larger than about 5x5 - must be a
            # numerical instability.  So as a hack I just drive any element < 10-4 to zero
            # statvec[statvec<0.0001] = 0
            fingerprint = statvec[:, np.newaxis] * (Anorm**0.5)
            fingerprints.append(fingerprint)

        return np.stack(fingerprints, axis=0)

    def transform(self, X):
        """Get state matrices and fingerprints from activation matrices"""
        As, Ppis = self.getStateMatrices(X)
        fprints = self.getFingerprints(As, Ppis)
        return [fprints, As, Ppis]

    def fit_transform(self, X, EW, NbSteps=1):
        """Fit HMM and return fingerprints et al."""
        self.fit(X, EW, NbSteps=1)
        return self.transform(X)
