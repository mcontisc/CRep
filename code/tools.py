"""
    Functions for handling the data.
"""

import networkx as nx
import numpy as np
import pandas as pd
import sktensor as skt


def import_data(dataset, ego='source', alter='target', force_dense=True, header=None):
    """
        Import data, i.e. the adjacency matrix, from a given folder.

        Return the NetworkX graph and its numpy adjacency matrix.

        Parameters
        ----------
        dataset : str
                  Path of the input file.
        ego : str
              Name of the column to consider as source of the edge.
        alter : str
                Name of the column to consider as target of the edge.
        force_dense : bool
                      If set to True, the algorithm is forced to consider a dense adjacency tensor.
        header : int
                 Row number to use as the column names, and the start of the data.

        Returns
        -------
        A : list
            List of MultiDiGraph NetworkX objects.
        B : ndarray/sptensor
            Graph adjacency tensor.
        B_T : None/sptensor
              Graph adjacency tensor (transpose).
        data_T_vals : None/ndarray
                      Array with values of entries A[j, i] given non-zero entry (i, j).
    """

    # read adjacency file
    df_adj = pd.read_csv(dataset, sep='\s+', header=header)
    print('{0} shape: {1}'.format(dataset, df_adj.shape))

    A = read_graph(df_adj=df_adj, ego=ego, alter=alter, noselfloop=True)

    nodes = list(A[0].nodes())

    # save the network in a tensor
    if force_dense:
        B, rw = build_B_from_A(A, nodes=nodes)
        B_T, data_T_vals = None, None
    else:
        B, B_T, data_T_vals, rw = build_sparse_B_from_A(A)

    print_graph_stat(A, rw)

    return A, B, B_T, data_T_vals


def read_graph(df_adj, ego='source', alter='target', noselfloop=True):
    """
        Create the graph by adding edges and nodes.
        It assumes that columns of layers are from l+2 (included) onwards.

        Return the list MultiDiGraph NetworkX objects.

        Parameters
        ----------
        df_adj : DataFrame
                 Pandas DataFrame object containing the edges of the graph.
        ego : str
              Name of the column to consider as source of the edge.
        alter : str
                Name of the column to consider as target of the edge.
        noselfloop : bool
                     If set to True, the algorithm removes the self-loops.

        Returns
        -------
        A : list
            List of MultiDiGraph NetworkX objects.
    """

    # build nodes
    egoID = df_adj[ego].unique()
    alterID = df_adj[alter].unique()
    nodes = list(set(egoID).union(set(alterID)))
    nodes.sort()

    L = df_adj.shape[1] - 2  # number of layers
    # build the NetworkX graph: create a list of graphs, as many graphs as there are layers
    A = [nx.MultiDiGraph() for _ in range(L)]
    # set the same set of nodes and order over all layers
    for l in range(L):
        A[l].add_nodes_from(nodes)

    for index, row in df_adj.iterrows():
        v1 = row[ego]
        v2 = row[alter]
        for l in range(L):
            if row[l + 2] > 0:
                if A[l].has_edge(v1, v2):
                    A[l][v1][v2][0]['weight'] += int(row[l + 2])  # the edge already exists -> no parallel edge created
                else:
                    A[l].add_edge(v1, v2, weight=int(row[l + 2]))

    # remove self-loops
    if noselfloop:
        for l in range(L):
            A[l].remove_edges_from(list(nx.selfloop_edges(A[l])))

    return A


def print_graph_stat(A, rw):
    """
        Print the statistics of the graph A.

        Parameters
        ----------
        A : list
            List of MultiDiGraph NetworkX objects.
        rw : list
             List whose elements are reciprocity (considering the weights of the edges) values, one per each layer.
    """

    L = len(A)
    N = A[0].number_of_nodes()
    print('Number of nodes =', N)
    print('Number of layers =', L)

    print('Number of edges and average degree in each layer:')
    for l in range(L):
        E = A[l].number_of_edges()
        k = 2 * float(E) / float(N)
        M = np.sum([d['weight'] for u, v, d in list(A[l].edges(data=True))])
        kW = 2 * float(M) / float(N)

        print(f'E[{l}] = {E} - <k> = {np.round(k, 3)}')
        print(f'M[{l}] = {M} - <k_weighted> = {np.round(kW, 3)}')
        print(f'Reciprocity (networkX) = {np.round(nx.reciprocity(A[l]), 3)}')
        print(f'Reciprocity (intended as the proportion of bi-directional edges over the unordered pairs) = '
              f'{np.round(reciprocal_edges(A[l]), 3)}')
        print(f'Reciprocity (considering the weights of the edges) = {np.round(rw[l], 3)}')


def build_B_from_A(A, nodes=None):
    """
        Create the numpy adjacency tensor of a networkX graph.

        Parameters
        ----------
        A : list
            List of MultiDiGraph NetworkX objects.
        nodes : list
                List of nodes IDs.

        Returns
        -------
        B : ndarray
            Graph adjacency tensor.
        rw : list
             List whose elements are reciprocity (considering the weights of the edges) values, one per each layer.
    """

    N = A[0].number_of_nodes()
    if nodes is None:
        nodes = list(A[0].nodes())
    B = np.empty(shape=[len(A), N, N])
    rw = []
    for l in range(len(A)):
        B[l, :, :] = nx.to_numpy_matrix(A[l], weight='weight', dtype=int, nodelist=nodes)
        rw.append(np.multiply(B[l], B[l].T).sum() / B[l].sum())

    return B, rw


def build_sparse_B_from_A(A):
    """
        Create the sptensor adjacency tensor of a networkX graph.

        Parameters
        ----------
        A : list
            List of MultiDiGraph NetworkX objects.

        Returns
        -------
        data : sptensor
               Graph adjacency tensor.
        data_T : sptensor
                 Graph adjacency tensor (transpose).
        v_T : ndarray
              Array with values of entries A[j, i] given non-zero entry (i, j).
        rw : list
             List whose elements are reciprocity (considering the weights of the edges) values, one per each layer.
    """

    N = A[0].number_of_nodes()
    L = len(A)
    rw = []

    d1 = np.array((), dtype='int64')
    d2, d2_T = np.array((), dtype='int64'), np.array((), dtype='int64')
    d3, d3_T = np.array((), dtype='int64'), np.array((), dtype='int64')
    v, vT, v_T = np.array(()), np.array(()), np.array(())
    for l in range(L):
        b = nx.to_scipy_sparse_matrix(A[l])
        b_T = nx.to_scipy_sparse_matrix(A[l]).transpose()
        rw.append(np.sum(b.multiply(b_T))/np.sum(b))
        nz = b.nonzero()
        nz_T = b_T.nonzero()
        d1 = np.hstack((d1, np.array([l] * len(nz[0]))))
        d2 = np.hstack((d2, nz[0]))
        d2_T = np.hstack((d2_T, nz_T[0]))
        d3 = np.hstack((d3, nz[1]))
        d3_T = np.hstack((d3_T, nz_T[1]))
        v = np.hstack((v, np.array([b[i, j] for i, j in zip(*nz)])))
        vT = np.hstack((vT, np.array([b_T[i, j] for i, j in zip(*nz_T)])))
        v_T = np.hstack((v_T, np.array([b[j, i] for i, j in zip(*nz)])))
    subs_ = (d1, d2, d3)
    subs_T_ = (d1, d2_T, d3_T)
    data = skt.sptensor(subs_, v, shape=(L, N, N), dtype=v.dtype)
    data_T = skt.sptensor(subs_T_, vT, shape=(L, N, N), dtype=vT.dtype)

    return data, data_T, v_T, rw


def reciprocal_edges(G):
    """
        Compute the proportion of bi-directional edges, by considering the unordered pairs.

        Parameters
        ----------
        G: MultiDigraph
           MultiDiGraph NetworkX object.

        Returns
        -------
        reciprocity: float
                     Reciprocity value, intended as the proportion of bi-directional edges over the unordered pairs.
    """

    n_all_edge = G.number_of_edges()
    n_undirected = G.to_undirected().number_of_edges()  # unique pairs of edges, i.e. edges in the undirected graph
    n_overlap_edge = (n_all_edge - n_undirected)  # number of undirected edges reciprocated in the directed network

    if n_all_edge == 0:
        raise nx.NetworkXError("Not defined for empty graphs.")

    reciprocity = float(n_overlap_edge) / float(n_undirected)

    return reciprocity


def can_cast(string):
    """
        Verify if one object can be converted to integer object.

        Parameters
        ----------
        string : int or float or str
                 Name of the node.

        Returns
        -------
        bool : bool
               If True, the input can be converted to integer object.
    """

    try:
        int(string)
        return True
    except ValueError:
        return False


def normalize_nonzero_membership(u):
    """
        Given a matrix, it returns the same matrix normalized by row.

        Parameters
        ----------
        u: ndarray
           Numpy Matrix.

        Returns
        -------
        The matrix normalized by row.
    """

    den1 = u.sum(axis=1, keepdims=True)
    nzz = den1 == 0.
    den1[nzz] = 1.

    return u / den1
