"""
    Class definition of CRep, the algorithm to perform inference in networks with reciprocity.
    The latent variables are related to community memberships and reciprocity value.
"""

from __future__ import print_function
import time
import sys
import sktensor as skt
import numpy as np
import pandas as pd
from termcolor import colored


class CRep:
    def __init__(self, N=100, L=1, K=2, undirected=False, initialization=0, rseed=0, inf=1e10, err_max=1e-8, err=0.01,
                 N_real=1, tolerance=0.0001, decision=10, max_iter=500, out_inference=False,
                 out_folder='../data/output/', end_file='.dat', assortative=False, eta0=None, fix_eta=False,
                 constrained=True, verbose=False, file_u='../data/input/u.dat', file_v='../data/input/v.dat',
                 file_w='../data/input/w.dat'):
        self.N = N  # number of nodes
        self.L = L  # number of layers
        self.K = K  # number of communities
        self.undirected = undirected  # flag to call the undirected network
        self.rseed = rseed  # random seed for the initialization
        self.inf = inf  # initial value of the pseudo log-likelihood
        self.err_max = err_max  # minimum value for the parameters
        self.err = err  # noise for the initialization
        self.N_real = N_real  # number of iterations with different random initialization
        self.tolerance = tolerance  # tolerance parameter for convergence
        self.decision = decision  # convergence parameter
        self.max_iter = max_iter  # maximum number of EM steps before aborting
        self.out_inference = out_inference  # flag for storing the inferred parameters
        self.out_folder = out_folder  # path for storing the output
        self.end_file = end_file  # output file suffix
        self.assortative = assortative  # if True, the network is assortative
        self.fix_eta = fix_eta  # if True, the eta parameter is fixed
        self.constrained = constrained  # if True, use the configuration with constraints on the updates
        self.verbose = verbose  # flag to print details
        self.input_u = file_u  # path of the input file u (when initialization=1)
        self.input_v = file_v  # path of the input file v (when initialization=1)
        self.input_w = file_w  # path of the input file w (when initialization=1)
        self.eta0 = eta0  # initial value for the reciprocity coefficient
        if initialization not in {0, 1}:  # indicator for choosing how to initialize u, v and w
            raise ValueError('The initialization parameter can be either 0 or 1. It is used as an indicator to '
                             'initialize the membership matrices u and v and the affinity matrix w. If it is 0, they '
                             'will be generated randomly, otherwise they will upload from file.')
        self.initialization = initialization
        if self.eta0 is not None:
            if (self.eta0 < 0) or (self.eta0 > 1):
                raise ValueError('The reciprocity coefficient eta0 has to be in [0, 1]!')
        if self.fix_eta:
            if self.eta0 is None:
                self.eta0 = 0.0

        if self.initialization == 1:
            dfU = pd.read_csv(self.input_u, sep='\s+')
            self.N, self.K = dfU.shape
            self.N_real = 1

        # values of the parameters used during the update
        self.u = np.zeros((self.N, self.K), dtype=float)  # out-going membership
        self.v = np.zeros((self.N, self.K), dtype=float)  # in-going membership
        self.eta = 0.  # reciprocity coefficient

        # values of the parameters in the previous iteration
        self.u_old = np.zeros((self.N, self.K), dtype=float)  # out-going membership
        self.v_old = np.zeros((self.N, self.K), dtype=float)  # in-going membership
        self.eta_old = 0.  # reciprocity coefficient

        # final values after convergence --> the ones that maximize the pseudo log-likelihood
        self.u_f = np.zeros((self.N, self.K), dtype=float)  # out-going membership
        self.v_f = np.zeros((self.N, self.K), dtype=float)  # in-going membership
        self.eta_f = 0.  # reciprocity coefficient

        # values of the affinity tensor
        if self.assortative:  # purely diagonal matrix
            self.w = np.zeros((self.L, self.K), dtype=float)
            self.w_old = np.zeros((self.L, self.K), dtype=float)
            self.w_f = np.zeros((self.L, self.K), dtype=float)
        else:
            self.w = np.zeros((self.L, self.K, self.K), dtype=float)
            self.w_old = np.zeros((self.L, self.K, self.K), dtype=float)
            self.w_f = np.zeros((self.L, self.K, self.K), dtype=float)

        if self.fix_eta:
            self.eta = self.eta_old = self.eta_f = self.eta0

    def fit(self, data, data_T, data_T_vals, nodes, mask=None):
        """
            Model directed networks by using a probabilistic generative model that assume community parameters and
            reciprocity coefficient. The inference is performed via EM algorithm.

            Parameters
            ----------
            data : ndarray/sptensor
                   Graph adjacency tensor.
            data_T: None/sptensor
                    Graph adjacency tensor (transpose).
            data_T_vals : None/ndarray
                          Array with values of entries A[j, i] given non-zero entry (i, j).
            nodes : list
                    List of nodes IDs.
            mask : ndarray
                   Mask for selecting the held out set in the adjacency tensor in case of cross-validation.

            Returns
            -------
            u_f : ndarray
                  Out-going membership matrix.
            v_f : ndarray
                  In-coming membership matrix.
            w_f : ndarray
                  Affinity tensor.
            eta_f : float
                    Reciprocity coefficient.
            maxL : float
                   Maximum pseudo log-likelihood.
            final_it : int
                       Total number of iterations.
        """

        maxL = -self.inf  # initialization of the maximum pseudo log-likelihood

        if data_T is None:
            E = np.sum(data)  # weighted sum of edges (needed in the denominator of eta)
            data_T = np.einsum('aij->aji', data)
            data_T_vals = get_item_array_from_subs(data_T, data.nonzero())
            # pre-processing of the data to handle the sparsity
            data = preprocess(data)
            data_T = preprocess(data_T)
        else:
            E = np.sum(data.vals)

        # save the indexes of the nonzero entries
        if isinstance(data, skt.dtensor):
            subs_nz = data.nonzero()
        elif isinstance(data, skt.sptensor):
            subs_nz = data.subs

        rng = np.random.RandomState(self.rseed)

        for r in range(self.N_real):

            self._initialize(rng=np.random.RandomState(self.rseed))

            self._update_old_variables()
            self._update_cache(data, data_T_vals, subs_nz)

            # convergence local variables
            coincide, it = 0, 0
            convergence = False
            loglik = self.inf

            if self.verbose:
                print(f'Updating realization {r} ...', end=' ')
            time_start = time.time()
            # --- single step iteration update ---
            while not convergence and it < self.max_iter:
                # main EM update: updates memberships and calculates max difference new vs old

                delta_u, delta_v, delta_w, delta_eta = self._update_em(data, data_T_vals, subs_nz, denominator=E)
                it, loglik, coincide, convergence = self._check_for_convergence(data, it, loglik, coincide, convergence,
                                                                                data_T=data_T, mask=mask)
            if self.verbose:
                print('done!')
                print(f'Nreal = {r} - Pseudo Loglikelihood = {loglik} - iterations = {it} - '
                      f'time = {np.round(time.time() - time_start, 2)} seconds')

            if maxL < loglik:
                self._update_optimal_parameters()
                maxL = loglik
                self.final_it = it
                conv = convergence
            self.rseed += rng.randint(100000000)
        # end cycle over realizations

        self.maxPSL = maxL
        if self.final_it == self.max_iter and not conv:
            # convergence not reaches
            print(colored('Solution failed to converge in {0} EM steps!'.format(self.max_iter), 'blue'))

        if self.out_inference:
            self.output_results(nodes)

        return self.u_f, self.v_f, self.w_f, self.eta_f, maxL

    def _initialize(self, rng=None):
        """
            Random initialization of the parameters u, v, w, eta.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)

        if self.eta0 is not None:
            self.eta = self.eta0
        else:
            self._randomize_eta(rng)

        if self.initialization == 0:
            if self.verbose:
                print('u, v and w are initialized randomly.')
            self._randomize_w(rng=rng)
            self._randomize_u_v(rng=rng)

        elif self.initialization == 1:
            if self.verbose:
                print('u, v and w are initialized using the input files:')
                print(self.input_u)
                print(self.input_v)
                print(self.input_w)
            self._initialize_u(self.input_u)
            self._initialize_v(self.input_v)
            self._initialize_w(self.input_w)

    def _randomize_eta(self, rng=None):
        """
            Generate a random number in (0, 1.).

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)
        self.eta = rng.random_sample(1)[0]

    def _randomize_w(self, rng):
        """
            Assign a random number in (0, 1.) to each entry of the affinity tensor w.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)
        for i in range(self.L):
            for k in range(self.K):
                if self.assortative:
                    self.w[i, k] = rng.random_sample(1)
                else:
                    for q in range(k, self.K):
                        if q == k:
                            self.w[i, k, q] = rng.random_sample(1)
                        else:
                            self.w[i, k, q] = self.w[i, q, k] = self.err * rng.random_sample(1)

    def _randomize_u_v(self, rng=None):
        """
            Assign a random number in (0, 1.) to each entry of the membership matrices u and v, and normalize each row.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)
        self.u = rng.random_sample(self.u.shape)
        row_sums = self.u.sum(axis=1)
        self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        if not self.undirected:
            self.v = rng.random_sample(self.v.shape)
            row_sums = self.v.sum(axis=1)
            self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
        else:
            self.v = self.u

    def _initialize_u(self, infile_name):
        """
            Initialize out-going membership matrix u from file.

            Parameters
            ----------
            infile_name : str
                          Path of the input file.
        """

        with open(infile_name, 'rb') as f:
            dfU = pd.read_csv(f, sep='\s+')
            self.u = dfU.values

        max_entry = np.max(self.u)
        self.u += max_entry * self.err * np.random.random_sample(self.u.shape)

    def _initialize_v(self, infile_name):
        """
            Initialize in-coming membership matrix v from file.

            Parameters
            ----------
            infile_name : str
                          Path of the input file.
        """

        if self.undirected:
            self.v = self.u
        else:
            with open(infile_name, 'rb') as f:
                dfV = pd.read_csv(f, sep='\s+')
                self.v = dfV.values

            max_entry = np.max(self.v)
            self.v += max_entry * self.err * np.random.random_sample(self.v.shape)

    def _initialize_w(self, infile_name):
        """
            Initialize affinity tensor w from file.

            Parameters
            ----------
            infile_name : str
                          Path of the input file.
        """

        with open(infile_name, 'rb') as f:
            dfW = pd.read_csv(f, sep='\s+')
            if self.assortative:
                self.w = np.diag(dfW)[np.newaxis, :].copy()
            else:
                self.w = dfW.values[np.newaxis, :, :]

        max_entry = np.max(self.w)
        self.w += max_entry * self.err * np.random.random_sample(self.w.shape)

    def _update_old_variables(self):
        """
            Update values of the parameters in the previous iteration.
        """

        self.u_old[self.u > 0] = np.copy(self.u[self.u > 0])
        self.v_old[self.v > 0] = np.copy(self.v[self.v > 0])
        self.w_old[self.w > 0] = np.copy(self.w[self.w > 0])
        self.eta_old = np.copy(self.eta)

    def _update_cache(self, data, data_T_vals, subs_nz):
        """
            Update the cache used in the em_update.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            data_T_vals : ndarray
                          Array with values of entries A[j, i] given non-zero entry (i, j).
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
        """

        self.lambda0_nz = self._lambda0_nz(subs_nz, self.u, self.v, self.w)
        self.M_nz = self.lambda0_nz + self.eta * data_T_vals
        self.M_nz[self.M_nz == 0] = 1
        if isinstance(data, skt.dtensor):
            self.data_M_nz = data[subs_nz] / self.M_nz
        elif isinstance(data, skt.sptensor):
            self.data_M_nz = data.vals / self.M_nz

    def _lambda0_nz(self, subs_nz, u, v, w):
        """
            Compute the mean lambda0_ij for only non-zero entries.

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            u : ndarray
                Out-going membership matrix.
            v : ndarray
                In-coming membership matrix.
            w : ndarray
                Affinity tensor.

            Returns
            -------
            nz_recon_I : ndarray
                         Mean lambda0_ij for only non-zero entries.
        """

        if not self.assortative:
            nz_recon_IQ = np.einsum('Ik,Ikq->Iq', u[subs_nz[1], :], w[subs_nz[0], :, :])
        else:
            nz_recon_IQ = np.einsum('Ik,Ik->Ik', u[subs_nz[1], :], w[subs_nz[0], :])
        nz_recon_I = np.einsum('Iq,Iq->I', nz_recon_IQ, v[subs_nz[2], :])

        return nz_recon_I

    def _update_em(self, data, data_T_vals, subs_nz, denominator=None):
        """
            Update parameters via EM procedure.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            data_T_vals : ndarray
                          Array with values of entries A[j, i] given non-zero entry (i, j).
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            denominator : float
                          Denominator used in the update of the eta parameter.

            Returns
            -------
            d_u : float
                  Maximum distance between the old and the new membership matrix u.
            d_v : float
                  Maximum distance between the old and the new membership matrix v.
            d_w : float
                  Maximum distance between the old and the new affinity tensor w.
            d_eta : float
                    Maximum distance between the old and the new reciprocity coefficient eta.
        """

        if not self.fix_eta:
            d_eta = self._update_eta(data, data_T_vals, denominator=denominator)
        else:
            d_eta = 0.
        self._update_cache(data, data_T_vals, subs_nz)

        d_u = self._update_U(subs_nz)
        self._update_cache(data, data_T_vals, subs_nz)

        if self.undirected:
            self.v = self.u
            self.v_old = self.v
            d_v = d_u
        else:
            d_v = self._update_V(subs_nz)
        self._update_cache(data, data_T_vals, subs_nz)

        if self.initialization != 1:
            if not self.assortative:
                d_w = self._update_W(subs_nz)
            else:
                d_w = self._update_W_assortative(subs_nz)
        else:
            d_w = 0
        self._update_cache(data, data_T_vals, subs_nz)

        return d_u, d_v, d_w, d_eta

    def _update_eta(self, data, data_T_vals, denominator=None):
        """
            Update reciprocity coefficient eta.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            data_T_vals : ndarray
                          Array with values of entries A[j, i] given non-zero entry (i, j).
            denominator : float
                          Denominator used in the update of the eta parameter.

            Returns
            -------
            dist_eta : float
                       Maximum distance between the old and the new reciprocity coefficient eta.
        """

        if denominator is None:
            Deta = data.sum()
        else:
            Deta = denominator

        self.eta *= (self.data_M_nz * data_T_vals).sum() / Deta

        dist_eta = abs(self.eta - self.eta_old)
        self.eta_old = np.copy(self.eta)

        return dist_eta

    def _update_U(self, subs_nz):
        """
            Update out-going membership matrix.

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.

            Returns
            -------
            dist_u : float
                     Maximum distance between the old and the new membership matrix u.
        """

        self.u = self.u_old * self._update_membership(subs_nz, self.u, self.v, self.w, 1)

        if not self.constrained:
            Du = np.einsum('iq->q', self.v)
            if not self.assortative:
                w_k = np.einsum('akq->kq', self.w)
                Z_uk = np.einsum('q,kq->k', Du, w_k)
            else:
                w_k = np.einsum('ak->k', self.w)
                Z_uk = np.einsum('k,k->k', Du, w_k)
            non_zeros = Z_uk > 0.
            self.u[:, Z_uk == 0] = 0.
            self.u[:, non_zeros] /= Z_uk[np.newaxis, non_zeros]
        else:
            row_sums = self.u.sum(axis=1)
            self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        low_values_indices = self.u < self.err_max  # values are too low
        self.u[low_values_indices] = 0.  # and set to 0.

        dist_u = np.amax(abs(self.u - self.u_old))
        self.u_old = np.copy(self.u)

        return dist_u

    def _update_V(self, subs_nz):
        """
            Update in-coming membership matrix.
            Same as _update_U but with:
            data <-> data_T
            w <-> w_T
            u <-> v

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.

            Returns
            -------
            dist_v : float
                     Maximum distance between the old and the new membership matrix v.
        """

        self.v *= self._update_membership(subs_nz, self.u, self.v, self.w, 2)

        if not self.constrained:
            Dv = np.einsum('iq->q', self.u)
            if not self.assortative:
                w_k = np.einsum('aqk->qk', self.w)
                Z_vk = np.einsum('q,qk->k', Dv, w_k)
            else:
                w_k = np.einsum('ak->k', self.w)
                Z_vk = np.einsum('k,k->k', Dv, w_k)
            non_zeros = Z_vk > 0
            self.v[:, Z_vk == 0] = 0.
            self.v[:, non_zeros] /= Z_vk[np.newaxis, non_zeros]
        else:
            row_sums = self.v.sum(axis=1)
            self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        low_values_indices = self.v < self.err_max  # values are too low
        self.v[low_values_indices] = 0.  # and set to 0.

        dist_v = np.amax(abs(self.v - self.v_old))
        self.v_old = np.copy(self.v)

        return dist_v

    def _update_W(self, subs_nz):
        """
            Update affinity tensor.

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.

            Returns
            -------
            dist_w : float
                     Maximum distance between the old and the new affinity tensor w.
        """

        sub_w_nz = self.w.nonzero()
        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Iq->Ikq', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis, np.newaxis] * UV
        for a, k, q in zip(*sub_w_nz):
            uttkrp_DKQ[:, k, q] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k, q], minlength=self.L)

        self.w *= uttkrp_DKQ

        Z = np.einsum('k,q->kq', self.u.sum(axis=0), self.v.sum(axis=0))[np.newaxis, :, :]
        non_zeros = Z > 0
        self.w[non_zeros] /= Z[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_W_assortative(self, subs_nz):
        """
            Update affinity tensor (assuming assortativity).

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.

            Returns
            -------
            dist_w : float
                     Maximum distance between the old and the new affinity tensor w.
        """

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis] * UV
        for k in range(self.K):
            uttkrp_DKQ[:, k] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k], minlength=self.L)

        self.w *= uttkrp_DKQ

        Z = ((self.u_old.sum(axis=0)) * (self.v_old.sum(axis=0)))[np.newaxis, :]
        non_zeros = Z > 0
        self.w[non_zeros] /= Z[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_membership(self, subs_nz, u, v, w, m):
        """
            Return the Khatri-Rao product (sparse version) used in the update of the membership matrices.

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            u : ndarray
                Out-going membership matrix.
            v : ndarray
                In-coming membership matrix.
            w : ndarray
                Affinity tensor.
            m : int
                Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
                works with the matrix u; if 2 it works with v.

            Returns
            -------
            uttkrp_DK : ndarray
                        Matrix which is the result of the matrix product of the unfolding of the tensor and the
                        Khatri-Rao product of the membership matrix.
        """

        if not self.assortative:
            uttkrp_DK = sp_uttkrp(self.data_M_nz, subs_nz, m, u, v, w)
        else:
            uttkrp_DK = sp_uttkrp_assortative(self.data_M_nz, subs_nz, m, u, v, w)

        return uttkrp_DK

    def _check_for_convergence(self, data, it, loglik, coincide, convergence, data_T=None, mask=None):
        """
            Check for convergence by using the pseudo log-likelihood values.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            it : int
                 Number of iteration.
            loglik : float
                     Pseudo log-likelihood value.
            coincide : int
                       Number of time the update of the pseudo log-likelihood respects the tolerance.
            convergence : bool
                          Flag for convergence.
            data_T : sptensor/dtensor
                     Graph adjacency tensor (transpose).
            mask : ndarray
                   Mask for selecting the held out set in the adjacency tensor in case of cross-validation.

            Returns
            -------
            it : int
                 Number of iteration.
            loglik : float
                     Log-likelihood value.
            coincide : int
                       Number of time the update of the pseudo log-likelihood respects the tolerance.
            convergence : bool
                          Flag for convergence.
        """

        if it % 10 == 0:
            old_L = loglik
            loglik = self.__PSLikelihood(data, self.eta, data_T=data_T, mask=mask)
            if abs(loglik - old_L) < self.tolerance:
                coincide += 1
            else:
                coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, loglik, coincide, convergence

    def __PSLikelihood(self, data, eta, data_T, mask=None):
        """
            Compute the pseudo log-likelihood of the data.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            data_T : sptensor/dtensor
                     Graph adjacency tensor (transpose).
            mask : ndarray
                   Mask for selecting the held out set in the adjacency tensor in case of cross-validation.

            Returns
            -------
            l : float
                Pseudo log-likelihood value.
        """

        self.lambda0_ija = self._lambda0_full(self.u, self.v, self.w)

        if mask is not None:
            sub_mask_nz = mask.nonzero()
            if isinstance(data, skt.dtensor):
                l = -self.lambda0_ija[sub_mask_nz].sum() - eta * data_T[sub_mask_nz].sum()
            elif isinstance(data, skt.sptensor):
                l = -self.lambda0_ija[sub_mask_nz].sum() - eta * data_T.toarray()[sub_mask_nz].sum()
        else:
            if isinstance(data, skt.dtensor):
                l = -self.lambda0_ija.sum() - eta * data_T.sum()
            elif isinstance(data, skt.sptensor):
                l = -self.lambda0_ija.sum() - eta * data_T.vals.sum()
        logM = np.log(self.M_nz)
        if isinstance(data, skt.dtensor):
            Alog = data[data.nonzero()] * logM
        elif isinstance(data, skt.sptensor):
            Alog = data.vals * logM

        l += Alog.sum()

        if np.isnan(l):
            print("PSLikelihood is NaN!!!!")
            sys.exit(1)
        else:
            return l

    def _lambda0_full(self, u, v, w):
        """
            Compute the mean lambda0 for all entries.

            Parameters
            ----------
            u : ndarray
                Out-going membership matrix.
            v : ndarray
                In-coming membership matrix.
            w : ndarray
                Affinity tensor.

            Returns
            -------
            M : ndarray
                Mean lambda0 for all entries.
        """

        if w.ndim == 2:
            M = np.einsum('ik,jk->ijk', u, v)
            M = np.einsum('ijk,ak->aij', M, w)
        else:
            M = np.einsum('ik,jq->ijkq', u, v)
            M = np.einsum('ijkq,akq->aij', M, w)
        return M

    def _update_optimal_parameters(self):
        """
            Update values of the parameters after convergence.
        """

        self.u_f = np.copy(self.u)
        self.v_f = np.copy(self.v)
        self.w_f = np.copy(self.w)
        self.eta_f = np.copy(self.eta)

    def output_results(self, nodes):
        """
            Output results.

            Parameters
            ----------
            nodes : list
                    List of nodes IDs.
        """

        outfile = self.out_folder + 'theta' + self.end_file
        np.savez_compressed(outfile + '.npz', u=self.u_f, v=self.v_f, w=self.w_f, eta=self.eta_f, max_it=self.final_it,
                            maxPSL=self.maxPSL, nodes=nodes)
        print(f'\nInferred parameters saved in: {outfile + ".npz"}')
        print('To load: theta=np.load(filename), then e.g. theta["u"]')


def sp_uttkrp(vals, subs, m, u, v, w):
    """
        Compute the Khatri-Rao product (sparse version).

        Parameters
        ----------
        vals : ndarray
               Values of the non-zero entries.
        subs : tuple
               Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must be
               equal to the dimension of tensor.
        m : int
            Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
            works with the matrix u; if 2 it works with v.
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.

        Returns
        -------
        out : ndarray
              Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao product
              of the membership matrix.
    """

    if m == 1:
        D, K = u.shape
        out = np.zeros_like(u)
    elif m == 2:
        D, K = v.shape
        out = np.zeros_like(v)

    for k in range(K):
        tmp = vals.copy()
        if m == 1:  # we are updating u
            tmp *= (w[subs[0], k, :].astype(tmp.dtype) * v[subs[2], :].astype(tmp.dtype)).sum(axis=1)
        elif m == 2:  # we are updating v
            tmp *= (w[subs[0], :, k].astype(tmp.dtype) * u[subs[1], :].astype(tmp.dtype)).sum(axis=1)
        out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)

    return out


def sp_uttkrp_assortative(vals, subs, m, u, v, w):
    """
        Compute the Khatri-Rao product (sparse version) with the assumption of assortativity.

        Parameters
        ----------
        vals : ndarray
               Values of the non-zero entries.
        subs : tuple
               Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must be
               equal to the dimension of tensor.
        m : int
            Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
            works with the matrix u; if 2 it works with v.
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.

        Returns
        -------
        out : ndarray
              Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao product
              of the membership matrix.
    """

    if m == 1:
        D, K = u.shape
        out = np.zeros_like(u)
    elif m == 2:
        D, K = v.shape
        out = np.zeros_like(v)

    for k in range(K):
        tmp = vals.copy()
        if m == 1:  # we are updating u
            tmp *= w[subs[0], k].astype(tmp.dtype) * v[subs[2], k].astype(tmp.dtype)
        elif m == 2:  # we are updating v
            tmp *= w[subs[0], k].astype(tmp.dtype) * u[subs[1], k].astype(tmp.dtype)
        out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)

    return out


def get_item_array_from_subs(A, ref_subs):
    """
        Get values of ref_subs entries of a dense tensor.
        Output is a 1-d array with dimension = number of non zero entries.
    """

    return np.array([A[a, i, j] for a, i, j in zip(*ref_subs)])


def preprocess(X):
    """
        Pre-process input data tensor.
        If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.

        Parameters
        ----------
        X : ndarray
            Input data (tensor).

        Returns
        -------
        X : sptensor/dtensor
            Pre-processed data. If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.
    """

    if not X.dtype == np.dtype(int).type:
        X = X.astype(int)
    if isinstance(X, np.ndarray) and is_sparse(X):
        X = sptensor_from_dense_array(X)
    else:
        X = skt.dtensor(X)

    return X


def is_sparse(X):
    """
        Check whether the input tensor is sparse.
        It implements a heuristic definition of sparsity. A tensor is considered sparse if:
        given
        M = number of modes
        S = number of entries
        I = number of non-zero entries
        then
        N > M(I + 1)

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        Boolean flag: true if the input tensor is sparse, false otherwise.
    """

    M = X.ndim
    S = X.size
    I = X.nonzero()[0].size

    return S > (I + 1) * M


def sptensor_from_dense_array(X):
    """
        Create an sptensor from a ndarray or dtensor.
        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        sptensor from a ndarray or dtensor.
    """

    subs = X.nonzero()
    vals = X[subs]

    return skt.sptensor(subs, vals, shape=X.shape, dtype=X.dtype)
