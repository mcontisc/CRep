"""
    Functions used in the k-fold cross-validation procedure.
"""

import CRep as CREP
import numpy as np
from sklearn import metrics
import yaml


def PSloglikelihood(B, u, v, w, eta, mask=None):
    """
        Compute the pseudo log-likelihood of the data.

        Parameters
        ----------
        B : ndarray
            Graph adjacency tensor.
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.
        eta : float
              Reciprocity coefficient.
        mask : ndarray
               Mask for selecting the held out set in the adjacency tensor in case of cross-validation.

        Returns
        -------
        Pseudo log-likelihood value.
    """

    if mask is None:
        M = _lambda0_full(u, v, w)
        M += (eta * B[0, :, :].T)[np.newaxis, :, :]
        logM = np.zeros(M.shape)
        logM[M > 0] = np.log(M[M > 0])
        return (B * logM).sum() - M.sum()
    else:
        M = _lambda0_full(u, v, w)[mask > 0]
        M += (eta * B[0, :, :].T)[np.newaxis, :, :][mask > 0]
        logM = np.zeros(M.shape)
        logM[M > 0] = np.log(M[M > 0])
        return (B[mask > 0] * logM).sum() - M.sum()


def _lambda0_full(u, v, w):
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


def transpose_ij(M):
    """
        Compute the transpose of a matrix.

        Parameters
        ----------
        M : ndarray
            Numpy matrix.

        Returns
        -------
        Transpose of the matrix.
    """

    return np.einsum('aij->aji', M)


def calculate_expectation(u, v, w, eta=0.0):
    """
        Compute the expectations, e.g. the parameters of the marginal distribution m_{ij}.

        Parameters
        ----------
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.
        eta : float
              Reciprocity coefficient.

        Returns
        -------
        M : ndarray
            Matrix whose elements are m_{ij}.
    """

    lambda0 = _lambda0_full(u, v, w)
    lambda0T = transpose_ij(lambda0)
    M = (lambda0 + eta * lambda0T) / (1. - eta * eta)

    return M


def calculate_conditional_expectation(B, u, v, w, eta=0.0, mean=None):
    """
        Compute the conditional expectations, e.g. the parameters of the conditional distribution lambda_{ij}.

        Parameters
        ----------
        B : ndarray
            Graph adjacency tensor.
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.
        eta : float
              Reciprocity coefficient.
        mean : ndarray
               Matrix with mean entries.

        Returns
        -------
        Matrix whose elements are lambda_{ij}.
    """

    if mean is None:
        return _lambda0_full(u, v, w) + eta * transpose_ij(B)  # conditional expectation (knowing A_ji)
    else:
        return _lambda0_full(u, v, w) + eta * transpose_ij(mean)


def calculate_AUC(pred, data0, mask=None):
    """
        Return the AUC of the link prediction. It represents the probability that a randomly chosen missing connection
        (true positive) is given a higher score by our method than a randomly chosen pair of unconnected vertices
        (true negative).

        Parameters
        ----------
        pred : ndarray
               Inferred values.
        data0 : ndarray
                Given values.
        mask : ndarray
               Mask for selecting a subset of the adjacency tensor.

        Returns
        -------
        AUC value.
    """

    data = (data0 > 0).astype('int')
    if mask is None:
        fpr, tpr, thresholds = metrics.roc_curve(data.flatten(), pred.flatten())
    else:
        fpr, tpr, thresholds = metrics.roc_curve(data[mask > 0], pred[mask > 0])

    return metrics.auc(fpr, tpr)


def shuffle_indices_all_matrix(N, L, rseed=10):
    """
        Shuffle the indices of the adjacency tensor.

        Parameters
        ----------
        N : int
            Number of nodes.
        L : int
            Number of layers.
        rseed : int
                Random seed.

        Returns
        -------
        indices : ndarray
                  Indices in a shuffled order.
    """

    n_samples = int(N * N)
    indices = [np.arange(n_samples) for _ in range(L)]
    rng = np.random.RandomState(rseed)
    for l in range(L):
        rng.shuffle(indices[l])

    return indices


def extract_mask_kfold(indices, N, fold=0, NFold=5):
    """
        Extract a non-symmetric mask using KFold cross-validation. It contains pairs (i,j) but possibly not (j,i).
        KFold means no train/test sets intersect across the K folds.

        Parameters
        ----------
        indices : ndarray
                  Indices of the adjacency tensor in a shuffled order.
        N : int
            Number of nodes.
        fold : int
               Current fold.
        NFold : int
                Number of total folds.

        Returns
        -------
        mask : ndarray
               Mask for selecting the held out set in the adjacency tensor.
    """

    L = len(indices)
    mask = np.zeros((L, N, N), dtype=bool)
    for l in range(L):
        n_samples = len(indices[l])
        test = indices[l][fold * (n_samples // NFold):(fold + 1) * (n_samples // NFold)]
        mask0 = np.zeros(n_samples, dtype=bool)
        mask0[test] = 1
        mask[l] = mask0.reshape((N, N))

    return mask


def fit_model(B, B_T, data_T_vals, nodes, N, L, algo, K, flag_conv, **conf):
    """
        Model directed networks by using a probabilistic generative model that assume community parameters and
        reciprocity coefficient. The inference is performed via EM algorithm.

        Parameters
        ----------
        B : ndarray
            Graph adjacency tensor.
        B_T : None/sptensor
              Graph adjacency tensor (transpose).
        data_T_vals : None/ndarray
                      Array with values of entries A[j, i] given non-zero entry (i, j).
        nodes : list
                List of nodes IDs.
        N : int
            Number of nodes.
        L : int
            Number of layers.
        algo : str
               Configuration to use (CRep, CRepnc, CRep0).
        K : int
            Number of communities.
        flag_conv : str
                    If 'log' the convergence is based on the log-likelihood values; if 'deltas' the convergence is
                    based on the differences in the parameters values. The latter is suggested when the dataset
                    is big (N > 1000 ca.).

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
        maxPSL : float
                 Maximum pseudo log-likelihood.
        mod : obj
              The CRep object.
    """

    # setting to run the algorithm
    with open(conf['out_folder'] + '/setting_' + algo + '.yaml', 'w') as f:
        yaml.dump(conf, f)

    mod = CREP.CRep(N=N, L=L, K=K, **conf)
    uf, vf, wf, nuf, maxPSL = mod.fit(data=B, data_T=B_T, data_T_vals=data_T_vals, flag_conv=flag_conv, nodes=nodes)

    return uf, vf, wf, nuf, maxPSL, mod


def calculate_opt_func(B, algo_obj=None, mask=None, assortative=False):
    """
        Compute the optimal value for the pseudo log-likelihood with the inferred parameters.

        Parameters
        ----------
        B : ndarray
            Graph adjacency tensor.
        algo_obj : obj
                   The CRep object.
        mask : ndarray
               Mask for selecting a subset of the adjacency tensor.
        assortative : bool
                      Flag to use an assortative mode.

        Returns
        -------
        Maximum pseudo log-likelihood value
    """

    B_test = B.copy()
    if mask is not None:
        B_test[np.logical_not(mask)] = 0.

    if not assortative:
        return PSloglikelihood(B, algo_obj.u_f, algo_obj.v_f, algo_obj.w_f, algo_obj.eta_f, mask=mask)
    else:
        L = B.shape[0]
        K = algo_obj.w_f.shape[-1]
        w = np.zeros((L, K, K))
        for l in range(L):
            w1 = np.zeros((K, K))
            np.fill_diagonal(w1, algo_obj.w_f[l])
            w[l, :, :] = w1.copy()
        return PSloglikelihood(B, algo_obj.u_f, algo_obj.v_f, w, algo_obj.eta_f, mask=mask)
