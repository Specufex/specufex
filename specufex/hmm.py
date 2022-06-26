import h5py
import numpy as np
from numpy.linalg import inv, matrix_power
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.special import psi
from scipy.stats import gamma

from .modelutils import SaveableModel


class BayesianHMM(SaveableModel):

    """BayesianHMM - class to fit hidden Markov model and calculate fingerprints.

    Arguments
    ---------
    num_pat: int
        Number of patterns in the NMF model. Used to initialize
    gain: numpy array
        The gain calculated for the NMF step.
        Accessible via BayesianNonparametricNMF.gain
        Perhaps we can get rid of these above 2 params at some point, then
        we could make the HMM model independent of the NMF model???
    num_state: int, default=15
        Number of states in the HMM.
    max_inner_ite: int, default=20
        Times to iterate in the forward-backward algorithm.
    Neff: int, default=100000
        Originally aproximately the number of samples, but seems
        to influence learning rate through lines 134 and 135 below
    verbose: int, default=0
        0 for no output, >0 for output
    """

    def __init__(
        self, num_pat, gain, num_state=15, max_inner_ite=20, Neff=100000, verbose=0
    ):

        self.num_state = num_state
        self._max_inner_ite = max_inner_ite
        self.verbose = verbose
        self.Neff = Neff

        self.num_pat = num_pat
        self.gain = gain

        self.tauPpi = 0.01 / self.num_state
        self.tauA = 0.1 / self.num_state
        self.tau1 = 0.1
        self.tau2 = 0.1

        self.B1 = (
            (self.Neff) * 1000 * gamma.rvs(np.ones((self.num_state, self.num_pat)), 1)
        )
        self.B2 = (self.Neff) * 1000 * 0.03 * np.ones((self.num_state, self.num_pat))

        self.step = 0

    def fit(self, V, resort_EB="energy"):
        """Fit the hidden Markov model.

        Internal variables for reference. ::

            dim:     rows in the EW matrix, also num_pat
            gain:    columns in the EW matrix
            X, V:    the activation matrix, num_pat(dim) x timesteps
            len_seq: timesteps in H matrix (and spectrogram)
            B:       num_state x dim - Gamma distributed matrix
            A:       num_state x num_state - Markov transition matrix, each row independently Dirichlet distributed
            lnPrH:   num_state x timesteps (len_seq)
            Ppi:     num_state - initial-state distribution

        Arguments
        ----------
        V: numpy array of float
            3-dimensional numpy array containing
            the activation matrices calculated in the NMF step.
            (n_matrices x num_pat x timesteps)
        verbose: int, default=0
            0 for no output, >0 for output
        resort_EB: None, str
            If None, do not sort the EB matrix (emissions matrix) rows (states).
            If "distance", do single-linkage hierarchical clustering on
            the pairwise distances between the rows, then order the rows by
            these distances. Currently, this doesn't produce the optimal global
            ordering in general.
            If "energy", sum each row and sort by sum in ascending order

        Returns
        --------
        None
            Saves the E1, E2, EB, and ElnB matrices to the object itself.

        """

        for step, Vi in enumerate(V):
            self.step += step
            n = int(np.round(1 + 10 * 0.75 ** (self.step - 1)))
            ElnB = psi(self.B1) - np.log(self.B2)
            EB = self.B1 / self.B2
            Bup1 = np.zeros_like(self.B1)
            Bup2 = np.zeros_like(self.B2)
            Atmp = np.zeros((self.num_state, self.num_state))

            len_seq = Vi.shape[1]  # number of timesteps
            # gain is the EW matrix summed along 0th dimension
            V2 = np.diag(self.gain) @ Vi  # scaling the right matrix, for HMM

            A = np.ones((self.num_state, self.num_state))
            Ppi = np.ones(self.num_state)
            for ite in range(self._max_inner_ite):
                expElnA = np.exp(
                    psi(A) - psi(A.sum(1))
                )  # convert each row to prob dist, find exp of ElnA
                expElnPi = np.exp(psi(Ppi) - psi(Ppi.sum()))  # same for Pi (int dist)
                lnPrH = ElnB @ V2 - EB.sum(1)[:, np.newaxis]
                lnPrH = (
                    lnPrH - np.max(lnPrH, axis=0)[np.newaxis, :]
                )  # convert to prob by dividing by max
                explnPrH = np.exp(lnPrH)

                # forward-backward algorithm
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
                Ppi = self.tauPpi + gam[:, 0]
                A = 0 * A + self.tauA
                for s in range(1, len_seq):
                    mat = expElnA * np.outer(
                        alpha[:, s - 1], beta[:, s] * explnPrH[:, s]
                    )
                    mat = mat / mat.sum()
                    A = A + mat
                    if ite == self._max_inner_ite - 1:
                        Bup1 = Bup1 + np.outer(gam[:, s], V2[:, s])
                        Bup2 = Bup2 + gam[:, s][:, np.newaxis]

            Atmp = Atmp + A
            Bup1 = Bup1 + np.outer(gam[:, 0], V2[:, 0])
            Bup2 = Bup2 + gam[:, 0][:, np.newaxis]

            rho = (250 / (1 + 5 * (n - 1)) + self.step) ** (-0.51)
            self.B1 = (1 - rho) * self.B1 + rho * (self.tau1 + (self.Neff / n) * Bup1)
            self.B2 = (1 - rho) * self.B2 + rho * (self.tau2 + (self.Neff / n) * Bup2)
            self.EB = self.B1 / self.B2
            self.ElnB = psi(self.B1) - np.log(self.B2)

            """if verbose > 0:
                print('step {}/{}'.format(step, len(V)))
            """

        # sort the EB matrix by pattern similarity
        # uses hierarchical clustering with single linkage
        if resort_EB == "distance":
            self._sort_EB_by_distance()
        if resort_EB == "energy":
            self._sort_EB_by_energy()

    def _getStateMatrices(self, V):
        """
        Create the state transition matrices for each activation matrix based
        on the HMM model fit previously.

        Arguments
        -----------
        V: 3 dimensional numpy array
            the array of V (activation) matrices to fit.

        Returns
        ----------
        numpy array
            The state matrices or each spectrogram in X.
        """

        As = []
        Ppis = []
        gams = []

        for Vi in V:

            len_seq = Vi.shape[1]
            V2 = np.diag(self.gain) @ Vi  # pull this outside of loop?

            A = np.ones((self.num_state, self.num_state))
            Ppi = np.ones(self.num_state)
            for _ in range(self._max_inner_ite):
                expElnA = np.exp(psi(A) - psi(A.sum(1)))
                expElnPi = np.exp(psi(Ppi) - psi(Ppi.sum()))
                lnPrH = self.ElnB @ V2 - self.EB.sum(1)[:, np.newaxis]
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
                Ppi = self.tauPpi + gam[:, 0]
                A = 0 * A + self.tauA
                for s in range(1, len_seq):
                    mat = expElnA * np.outer(
                        alpha[:, s - 1], beta[:, s] * explnPrH[:, s]
                    )
                    mat = mat / mat.sum()
                    A = A + mat
            gams.append(gam)
            As.append(A)
            Ppis.append(Ppi)

        gams = np.stack(gams)
        As = np.stack(As, axis=0)
        Ppis = np.squeeze(Ppis)

        return As, Ppis, gams

    def _getFingerprints(self, As, Ppis):
        """
        Calculate the state transition probabilities for each state matrix (A).

        Arguments
        -----------
        As: 3 dimensional numpy array
            the array of As (state transition) matrices to fit.

        Returns
        ----------
        numpy array
            The fingerprints for each A. This is the final feature vector for
            each spectrogram in X.

        TODO:
        -----
        Loop might be able to vectorized. `np.linalg.inv` broadcasts over
        batch (1st) dimension
        """
        eps = np.finfo(float).eps

        fingerprints = []
        # Pinorm and Anorm calcs are vectorized
        T = 100

        # TODO: This loop might be able to vectorized
        #  np.linalg.inv broadcasts over batch (1st) dimension
        for i in range(As.shape[0]):
            Ppi_len = len(Ppis[i])
            Ppi = Ppis[i] - 0.01 / Ppi_len

            A = As[i] - (0.1 - eps) / Ppi_len
            Pinorm = Ppi / Ppi.sum()
            Anorm = A / A.sum(1)[:, np.newaxis]
            Eye = np.eye(Ppi_len)
            a = Eye - T / (T + 1) * Anorm
            b = Eye - matrix_power((T / (T + 1) * Anorm), (T + 1))
            statvec = Pinorm.T @ inv(a) @ b
            statvec = statvec.T / statvec.sum()
            # numpy gives inaccurate (at least different from Matalb, which I think is correct)
            # matrix power results for high powers and matrices larger than about 5x5 - must be a
            # numerical instability. Not sure this actually affects results
            fingerprint = statvec[:, np.newaxis] * (Anorm**0.5)
            fingerprints.append(fingerprint)

        return np.stack(fingerprints, axis=0)

    def transform(self, V):
        """
        Create the fingerprints for each activation matrix based
        on the HMM model fit previously.

        Arguments
        -----------
        V: 3 dimensional numpy array
            the array of V (activation) matrices to fit.

        Returns
        ----------
        fprints : numpy array
            The fingerprints for each A. This is the final feature vector for
            each spectrogram in X.
        As : numpy array
            State transition matrices for each spectrogram, calculated from the HMM.
        gams : numpy array
            Stack of state-sequence matrices for each spectrogram. Columns in each
            matrix are timesteps, rows contain the probability that the system was in
            a particular HMM state.

        """
        As, Ppis, gams = self._getStateMatrices(V)
        fprints = self._getFingerprints(As, Ppis)
        return fprints, As, gams

    def fit_transform(self, V, verbose=0):
        """Fit HMM and return fingerprints in one step.

        Arguments
        ----------
        V: numpy array of float
            3-dimensional numpy array containing
            the activation matrices calculated in the NMF step.
            (n_matrices x num_pat x timesteps)

        Returns
        ----------
        numpy array
            The fingerprints for each A. This is the final feature vector for
            each activation matrix in V, and ultimately each spectrogram in X.
        """
        self.fit(V, verbose=verbose)
        return self.transform(V)

    def _sort_EB_by_distance(self):
        EB_dist = pdist(self.EB)
        link = linkage(EB_dist, method="single")
        EBidx = leaves_list(link)
        self._resort_B(EBidx)
        self.EB_dist = EB_dist  # distance matrix

    def _sort_EB_by_energy(self):
        EB_energy = self.EB.sum(axis=1)
        EBidx = np.argsort(EB_energy)
        self._resort_B(EBidx)
        self.EB_energy = EB_energy[EBidx]

    def _resort_B(self, EBidx):
        # print(f"EB shape {self.EB.shape}")
        # print(f"EB_index {EBidx}")
        self.EB = self.EB[EBidx, :]
        self.ElnB = self.ElnB[EBidx, :]
        self.B1 = self.B1[EBidx, :]
        self.B2 = self.B2[EBidx, :]

    @classmethod
    def load(cls, filename):
        """Load a saved model.

        Arguments
        ----------
        filename: string
            The name of an hdf5 file containing the parameters for a trained model

        Returns
        --------
        BayesianHMM
            object with the parameters from filename.
        """
        with h5py.File(filename, "r") as hf:
            # init the class with the EW matrix stored in the parameter file
            num_pat = hf["num_pat"][()]
            gain = hf["gain"][()]
            return cls(num_pat, gain)._load(filename)
