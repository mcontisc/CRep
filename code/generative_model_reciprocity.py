"""
    Class definition of the reciprocity generative model with the member functions required.
    It builds a directed, possibly weighted, network.
"""

import numpy as np
import networkx as nx
import pandas as pd
import scipy.sparse as sparse
import math
import tools as tl


class GM_reciprocity:
    def __init__(self, N, K, eta=0.5, k=3, ExpM=None, over=0., corr=0., seed=0, alpha=0.1, ag=0.1, beta=0.1,
                 Normalization=0, structure='assortative', end_file='', out_folder='../data/output/real_data/cv/',
                 output_parameters=False, output_adj=False, outfile_adj='None', verbose=False):
        self.N = N  # number of nodes
        self.K = K  # number of communities
        self.k = k  # average degree
        self.seed = seed  # random seed
        self.alpha = alpha  # parameter of the Dirichlet distribution
        self.ag = ag  # alpha parameter of the Gamma distribution
        self.beta = beta  # beta parameter of the Gamma distribution
        self.end_file = end_file  # output file suffix
        self.out_folder = out_folder  # path for storing the output
        self.output_parameters = output_parameters  # flag for storing the parameters
        self.output_adj = output_adj  # flag for storing the generated adjacency matrix
        self.outfile_adj = outfile_adj  # name for saving the adjacency matrix
        self.verbose = verbose  # flag to print details
        if (eta < 0) or (eta >= 1):  # reciprocity coefficient
            raise ValueError('The reciprocity coefficient eta has to be in [0, 1)!')
        self.eta = eta
        if ExpM is None:  # expected number of edges
            self.ExpM = int(self.N * self.k / 2.)
        else:
            self.ExpM = int(ExpM)
            self.k = 2 * self.ExpM / float(self.N)
        if (over < 0) or (over > 1):  # fraction of nodes with mixed membership
            raise ValueError('The over parameter has to be in [0, 1]!')
        self.over = over
        if (corr < 0) or (corr > 1):  # correlation between u and v synthetically generated
            raise ValueError('The correlation parameter corr has to be in [0, 1]!')
        self.corr = corr
        if Normalization not in {0, 1}:  # indicator for choosing how to generate the latent variables
            raise ValueError('The Normalization parameter can be either 0 or 1! It is used as an indicator for '
                             'generating the membership matrices u and v from a Dirichlet or a Gamma distribution, '
                             'respectively. It is used when there is overlapping.')
        self.Normalization = Normalization
        if structure not in {'assortative', 'disassortative'}:  # structure of the affinity matrix W
            raise ValueError('The structure of the affinity matrix w can be either assortative or disassortative!')
        self.structure = structure

    def reciprocity_planted_network(self, parameters=None):
        """
            Generate a directed, possibly weighted network by using the reciprocity generative model.
            Can be used to generate benchmarks for networks with reciprocity.

            Steps:
                1. Generate the latent variables.
                2. Extract A_ij entries (network edges) from a Poisson distribution;
                   its mean depends on the latent variables.

            Parameters
            ----------
            parameters: object
                        Latent variables u, v, w and eta.

            Returns
            -------
            G: MultiDigraph
               MultiDiGraph NetworkX object.
        """

        prng = np.random.RandomState(self.seed)  # set seed random number generator

        '''
        Set latent variables u, v, w
        '''

        if parameters is not None:
            self.u, self.v, self.w, self.eta = parameters
        else:
            # equal-size unmixed group membership
            size = int(self.N / self.K)
            self.u = np.zeros((self.N, self.K))
            self.v = np.zeros((self.N, self.K))
            for i in range(self.N):
                q = int(math.floor(float(i) / float(size)))
                if q == self.K:
                    self.u[i:, self.K - 1] = 1.
                    self.v[i:, self.K - 1] = 1.
                else:
                    for j in range(q * size, q * size + size):
                        self.u[j, q] = 1.
                        self.v[j, q] = 1.
            self.w = affinity_matrix(structure=self.structure, N=self.N, K=self.K, a=0.1, b=0.3)

            # in case of overlapping
            if self.over != 0.:
                overlapping = int(self.N * self.over)  # number of nodes belonging to more communities
                ind_over = np.random.randint(len(self.u), size=overlapping)
                if self.Normalization == 0:
                    # u and v from a Dirichlet distribution
                    self.u[ind_over] = prng.dirichlet(self.alpha * np.ones(self.K), overlapping)
                    self.v[ind_over] = self.corr * self.u[ind_over] + (1. - self.corr) * \
                                       prng.dirichlet(self.alpha * np.ones(self.K), overlapping)
                    if self.corr == 1.:
                        assert np.allclose(self.u, self.v)
                    if self.corr > 0:
                        self.v = tl.normalize_nonzero_membership(self.v)
                elif self.Normalization == 1:
                    # u and v from a Gamma distribution
                    self.u[ind_over] = prng.gamma(self.ag, 1. / self.beta, size=(overlapping, self.K))
                    self.v[ind_over] = self.corr * self.u[ind_over] + (1. - self.corr) * \
                                       prng.gamma(self.ag, 1. / self.beta, size=(overlapping, self.K))
                    self.u = tl.normalize_nonzero_membership(self.u)
                    self.v = tl.normalize_nonzero_membership(self.v)

        M0 = Exp_ija_matrix(self.u, self.v, self.w)  # whose elements are lambda0_{ij}
        np.fill_diagonal(M0, 0)

        c = (self.ExpM * (1. - self.eta)) / M0.sum()  # constant to enforce sparsity

        MM = (M0 + self.eta * transpose_ij(M0)) / (1. - self.eta * self.eta)  # whose elements are m_{ij}
        Mt = transpose_ij(MM)
        MM0 = M0.copy()  # to be not influenced by c_lambda

        if parameters is None:
            self.w *= c  # only w is impact by that, u and v have a constraint, their sum over k should sum to 1
        M0 *= c
        M0t = transpose_ij(M0)  # whose elements are lambda0_{ji}

        M = (M0 + self.eta * M0t) / (1. - self.eta * self.eta)  # whose elements are m_{ij}
        np.fill_diagonal(M, 0)

        rw = self.eta + ((MM0 * Mt + self.eta * Mt ** 2).sum() / MM.sum())  # expected reciprocity

        '''
        Generate network G (and adjacency matrix A) using the latent variables,
        with the generative model (A_ij,A_ji) ~ P(A_ij|u,v,w,eta) P(A_ji|A_ij,u,v,w,eta)
        '''

        G = nx.MultiDiGraph()
        for i in range(self.N):
            G.add_node(i)

        counter, totM = 0, 0,
        for i in range(self.N):
            for j in range(i + 1, self.N):
                r = prng.rand(1)[0]
                if r < 0.5:
                    A_ij = prng.poisson(M[i, j], 1)[0]  # draw A_ij from P(A_ij) = Poisson(m_ij)
                    if A_ij > 0:
                        G.add_edge(i, j, weight=A_ij)
                    lambda_ji = M0[j, i] + self.eta * A_ij
                    A_ji = prng.poisson(lambda_ji, 1)[0]  # draw A_ji from P(A_ji|A_ij) = Poisson(lambda0_ji + eta*A_ij)
                    if A_ji > 0:
                        G.add_edge(j, i, weight=A_ji)
                else:
                    A_ji = prng.poisson(M[j, i], 1)[0]  # draw A_ij from P(A_ij) = Poisson(m_ij)
                    if A_ji > 0:
                        G.add_edge(j, i, weight=A_ji)
                    lambda_ij = M0[i, j] + self.eta * A_ji
                    A_ij = prng.poisson(lambda_ij, 1)[0]  # draw A_ji from P(A_ji|A_ij) = Poisson(lambda0_ji + eta*A_ij)
                    if A_ij > 0:
                        G.add_edge(i, j, weight=A_ij)
                counter += 1
                totM += A_ij + A_ji

        nodes = list(G.nodes())

        # keep largest connected component
        Gc = max(nx.weakly_connected_components(G), key=len)
        nodes_to_remove = set(G.nodes()).difference(Gc)
        G.remove_nodes_from(list(nodes_to_remove))

        nodes = list(G.nodes())
        self.u = self.u[nodes]
        self.v = self.v[nodes]
        self.N = len(nodes)

        A = nx.to_scipy_sparse_matrix(G, nodelist=nodes, weight='weight')

        Sparsity_cof = np.round(2 * G.number_of_edges() / float(G.number_of_nodes()), 3)

        ave_w_deg = np.round(2 * totM / float(G.number_of_nodes()), 3)

        reciprocity_c = np.round(tl.reciprocal_edges(G), 3)

        if self.verbose:
            print(f'Number of links in the upper triangular matrix: {sparse.triu(A, k=1).nnz}\n'
                  f'Number of links in the lower triangular matrix: {sparse.tril(A, k=-1).nnz}')
            print(f'Sum of weights in the upper triangular matrix: {np.round(sparse.triu(A, k=1).sum(), 2)}\n'
                  f'Sum of weights in the lower triangular matrix: {np.round(sparse.tril(A, k=-1).sum(), 2)}\n'
                  f'Number of possible unordered pairs: {counter}')
            print(f'Removed {len(nodes_to_remove)} nodes, because not part of the largest connected component')
            print(f'Number of nodes: {G.number_of_nodes()} \n'
                  f'Number of edges: {G.number_of_edges()}')
            print(f'Average degree (2E/N): {Sparsity_cof}')
            print(f'Average weighted degree (2M/N): {ave_w_deg}')
            print(f'Expected reciprocity: {np.round(rw, 3)}')
            print(f'Reciprocity (intended as the proportion of bi-directional edges over the unordered pairs): '
                  f'{reciprocity_c}\n')

        if self.output_parameters:
            self.output_results(nodes)

        if self.output_adj:
            self.output_adjacency(G, outfile=self.outfile_adj)

        return G

    def planted_network_cond_independent(self, parameters=None):
        """
            Generate a directed, possibly weighted network without using reciprocity.
            It uses conditionally independent A_ij from a Poisson | (u,v,w).

            Parameters
            ----------
            parameters: object
                        Latent variables u, v and w.

            Returns
            -------
            G: MultiDigraph
               MultiDiGraph NetworkX object.
        """

        prng = np.random.RandomState(self.seed)  # set seed random number generator

        '''
        Set latent variables u,v,w
        '''

        if parameters is not None:
            self.u, self.v, self.w = parameters
        else:
            # equal-size unmixed group membership
            size = int(self.N / self.K)
            self.u = np.zeros((self.N, self.K))
            self.v = np.zeros((self.N, self.K))
            for i in range(self.N):
                q = int(math.floor(float(i) / float(size)))
                if q == self.K:
                    self.u[i:, self.K - 1] = 1.
                    self.v[i:, self.K - 1] = 1.
                else:
                    for j in range(q * size, q * size + size):
                        self.u[j, q] = 1.
                        self.v[j, q] = 1.
            self.w = affinity_matrix(structure=self.structure, N=self.N, K=self.K, a=0.1, b=0.3)

            # in case of overlapping
            if self.over != 0.:
                overlapping = int(self.N * self.over)  # number of nodes belonging to more communities
                ind_over = np.random.randint(len(self.u), size=overlapping)
                if self.Normalization == 0:
                    # u and v from a Dirichlet distribution
                    self.u[ind_over] = prng.dirichlet(self.alpha * np.ones(self.K), overlapping)
                    self.v[ind_over] = self.corr * self.u[ind_over] + (1. - self.corr) * \
                                       prng.dirichlet(self.alpha * np.ones(self.K), overlapping)
                    if self.corr == 1.:
                        assert np.allclose(self.u, self.v)
                    if self.corr > 0:
                        self.v = tl.normalize_nonzero_membership(self.v)
                elif self.Normalization == 1:
                    # u and v from a Gamma distribution
                    self.u[ind_over] = prng.gamma(self.ag, 1. / self.beta, size=(overlapping, self.K))
                    self.v[ind_over] = self.corr * self.u[ind_over] + (1. - self.corr) * \
                                       prng.gamma(self.ag, 1. / self.beta, size=(overlapping, self.K))
                    self.u = tl.normalize_nonzero_membership(self.u)
                    self.v = tl.normalize_nonzero_membership(self.v)

        M0 = Exp_ija_matrix(self.u, self.v, self.w)  # whose elements are lambda0_{ij}
        np.fill_diagonal(M0, 0)
        M0t = transpose_ij(M0)  # whose elements are lambda0_{ji}

        rw = (M0 * M0t).sum() / M0.sum()  # expected reciprocity

        c = self.ExpM / float(M0.sum())  # constant to enforce sparsity

        '''
        Generate network G (and adjacency matrix A) using the latent variable,
        with the generative model (A_ij) ~ P(A_ij|u,v,w) 
        '''

        G = nx.MultiDiGraph()
        for i in range(self.N):
            G.add_node(i)

        totM = 0
        for i in range(self.N):
            for j in range(self.N):
                if i != j:  # no self-loops
                    A_ij = prng.poisson(c * M0[i, j], 1)[0]  # draw A_ij from P(A_ij) = Poisson(c*m_ij)
                    if A_ij > 0:
                        G.add_edge(i, j, weight=A_ij)
                    totM += A_ij

        nodes = list(G.nodes())

        # keep largest connected component
        Gc = max(nx.weakly_connected_components(G), key=len)
        nodes_to_remove = set(G.nodes()).difference(Gc)
        G.remove_nodes_from(list(nodes_to_remove))

        nodes = list(G.nodes())
        self.u = self.u[nodes]
        self.v = self.v[nodes]
        self.N = len(nodes)

        A = nx.to_scipy_sparse_matrix(G, nodelist=nodes, weight='weight')

        Sparsity_cof = np.round(2 * G.number_of_edges() / float(G.number_of_nodes()), 3)

        ave_w_deg = np.round(2 * totM / float(G.number_of_nodes()), 3)

        reciprocity_c = np.round(tl.reciprocal_edges(G), 3)

        if self.verbose:
            print(f'Number of links in the upper triangular matrix: {sparse.triu(A, k=1).nnz}\n'
                  f'Number of links in the lower triangular matrix: {sparse.tril(A, k=-1).nnz}')
            print(f'Sum of weights in the upper triangular matrix: {np.round(sparse.triu(A, k=1).sum(), 2)}\n'
                  f'Sum of weights in the lower triangular matrix: {np.round(sparse.tril(A, k=-1).sum(), 2)}')
            print(f'Removed {len(nodes_to_remove)} nodes, because not part of the largest connected component')
            print(f'Number of nodes: {G.number_of_nodes()} \n'
                  f'Number of edges: {G.number_of_edges()}')
            print(f'Average degree (2E/N): {Sparsity_cof}')
            print(f'Average weighted degree (2M/N): {ave_w_deg}')
            print(f'Expected reciprocity: {np.round(rw, 3)}')
            print(f'Reciprocity (intended as the proportion of bi-directional edges over the unordered pairs): '
                  f'{reciprocity_c}\n')

        if self.output_parameters:
            self.output_results(nodes)

        if self.output_adj:
            self.output_adjacency(G, outfile=self.outfile_adj)

        return G

    def planted_network_reciprocity_only(self, p=None):
        """
            Generate a directed, possibly weighted network using only reciprocity.
            One of the directed-edges is generated with probability p, the other with eta*A_ji,
            i.e. as in Erdos-Renyi reciprocity.

            Parameters
            ----------
            p: float
               Probability to generate one of the directed-edge.

            Returns
            -------
            G: MultiDigraph
               MultiDiGraph NetworkX object.
        """

        prng = np.random.RandomState(self.seed)  # set seed random number generator

        if p is None:
            p = (1. - self.eta) * self.k * 0.5 / (self.N - 1.)

        '''
        Generate network G (and adjacency matrix A)
        '''

        G = nx.MultiDiGraph()
        for i in range(self.N):
            G.add_node(i)

        totM = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                A0 = prng.poisson(p, 1)[0]
                A1 = prng.poisson(p + A0, 1)[0]
                r = prng.rand(1)[0]
                if r < 0.5:
                    if A0 > 0:
                        G.add_edge(i, j, weight=A0)
                    if A1 > 0:
                        G.add_edge(j, i, weight=A1)
                else:
                    if A0 > 0:
                        G.add_edge(j, i, weight=A0)
                    if A1 > 0:
                        G.add_edge(i, j, weight=A1)
                totM += A0 + A1

        # keep largest connected component
        Gc = max(nx.weakly_connected_components(G), key=len)
        nodes_to_remove = set(G.nodes()).difference(Gc)
        G.remove_nodes_from(list(nodes_to_remove))

        nodes = list(G.nodes())
        self.N = len(nodes)

        A = nx.to_scipy_sparse_matrix(G, nodelist=nodes, weight='weight')

        Sparsity_cof = np.round(2 * G.number_of_edges() / float(G.number_of_nodes()), 3)

        ave_w_deg = np.round(2 * totM / float(G.number_of_nodes()), 3)

        reciprocity_c = np.round(tl.reciprocal_edges(G), 3)

        if self.verbose:
            print(f'Number of links in the upper triangular matrix: {sparse.triu(A, k=1).nnz}\n'
                  f'Number of links in the lower triangular matrix: {sparse.tril(A, k=-1).nnz}')
            print(f'Sum of weights in the upper triangular matrix: {np.round(sparse.triu(A, k=1).sum(), 2)}\n'
                  f'Sum of weights in the lower triangular matrix: {np.round(sparse.tril(A, k=-1).sum(), 2)}')
            print(f'Removed {len(nodes_to_remove)} nodes, because not part of the largest connected component')
            print(f'Number of nodes: {G.number_of_nodes()} \n'
                  f'Number of edges: {G.number_of_edges()}')
            print(f'Average degree (2E/N): {Sparsity_cof}')
            print(f'Average weighted degree (2M/N): {ave_w_deg}')
            print(f'Reciprocity (intended as the proportion of bi-directional edges over the unordered pairs): '
                  f'{reciprocity_c}\n')

        if self.output_adjacency:
            self.output_adjacency(G, outfile=self.outfile_adj)

        return G

    def output_results(self, nodes):
        """
            Output results in a compressed file.

            Parameters
            ----------
            nodes : list
                    List of nodes IDs.
        """

        output_parameters = self.out_folder + 'theta_gt' + str(self.seed) + self.end_file
        np.savez_compressed(output_parameters + '.npz', u=self.u, v=self.v, w=self.w, eta=self.eta, nodes=nodes)
        if self.verbose:
            print(f'Parameters saved in: {output_parameters}.npz')
            print('To load: theta=np.load(filename), then e.g. theta["u"]')

    def output_adjacency(self, G, outfile=None):
        """
            Output the adjacency matrix. Default format is space-separated .csv with 3 columns:
            node1 node2 weight

            Parameters
            ----------
            G: MultiDigraph
               MultiDiGraph NetworkX object.
            outfile: str
                     Name of the adjacency matrix.
        """

        if outfile is None:
            outfile = 'syn' + str(self.seed) + '_k' + str(int(self.k)) + '.dat'

        edges = list(G.edges(data=True))
        try:
            data = [[u, v, d['weight']] for u, v, d in edges]
        except:
            data = [[u, v, 1] for u, v, d in edges]

        df = pd.DataFrame(data, columns=['source', 'target', 'w'], index=None)
        df.to_csv(self.out_folder + outfile, index=False, sep=' ')
        if self.verbose:
            print(f'Adjacency matrix saved in: {self.out_folder + outfile}')


def Exp_ija_matrix(u, v, w):
    """
        Compute the mean lambda0_ij for all entries.

        Parameters
        ----------
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity matrix.

        Returns
        -------
        M : ndarray
            Mean lambda0_ij for all entries.
    """

    M = np.einsum('ik,jq->ijkq', u, v)
    M = np.einsum('ijkq,kq->ij', M, w)

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

    return np.einsum('ij->ji', M)


def affinity_matrix(structure='assortative', N=100, K=2, a=0.1, b=0.3):
    """
        Return the KxK affinity matrix w with probabilities between and within groups.

        Parameters
        ----------
        structure : string
                    Structure of the network, e.g. assortative, disassortative.
        N : int
            Number of nodes.
        K : int
            Number of communities.
        a : float
            Parameter for secondary probabilities.
        b : float
            Parameter for third probabilities.

        Returns
        -------
        p : ndarray
            Array with probabilities between and within groups. Element (k,q) gives the density of edges going from the
            nodes of group k to nodes of group q.
    """

    b *= a
    p1 = K / N
    if structure == 'assortative':
        p = p1 * a * np.ones((K, K))  # secondary-probabilities
        np.fill_diagonal(p, p1 * np.ones(K))  # primary-probabilities

    elif structure == 'disassortative':
        p = p1 * np.ones((K, K))  # primary-probabilities
        np.fill_diagonal(p, a * p1 * np.ones(K))  # secondary-probabilities

    # print(f'Affinity matrix w: \n{p}')

    return p
