"""Compressors"""
from collections import defaultdict
import numpy as np
from scipy import special, sparse
from tqdm import tqdm

from sknetwork.utils import get_degrees

from src.derivation import extension_csc


def mdl_graph(adjacency: sparse.csr_matrix) -> float:
    """Minimum description length for graph structure.

    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph

    Returns
    -------
        Minimum description length of graph structure."""
    n = adjacency.shape[0]

    if n == 0:
        return 0

    # nodes
    nodes_mdl = np.log2(n + 1)

    # edges
    degrees = adjacency.dot(np.ones(n))
    max_degree = np.max(degrees)
    edges_mdl = (n + 1) * np.log2(max_degree + 1) + \
        np.sum([np.log2(special.comb(n, deg)) for deg in degrees])

    return nodes_mdl + edges_mdl


def generation_complexity(adjacency: sparse.csr_matrix,
                          biadjacency: sparse.csr_matrix, n_attrs: int = 15,
                          n_iter: int = 300) -> dict:
    """Generation complexity of a graph.

    * Sample attributes w.r.t their degree
    * Retrieve induced subgraph
    * Compute description complexity of induced subgraph

    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    biadjacency: sparse.csr_matrix
        Feature matrix of the graph
    n_attrs: int (default=15)
        Maximum number of attributes
    n_iter: int (default=300)
        Maximum number of iterations.

    Returns
    ------
        Dictionary of description complexities of sampled graphs, according to
        their number of nodes.
    """
    print('Generation complexities for graph structure...')
    complexity_gen_graphs = defaultdict(list)

    # Correction if n_attrs > number of attributes in biadjacency
    if n_attrs > biadjacency.shape[1]:
        n_attrs = biadjacency.shape[1]

    attrs_degrees_prob = get_degrees(biadjacency, transpose=True) \
        / sum(get_degrees(biadjacency, transpose=True))
    attrs_indexes = np.arange(0, biadjacency.shape[1])
    biadjacency_csc = biadjacency.tocsc()

    for _ in tqdm(range(n_iter)):
        for num_a in range(n_attrs):
            # Randomly select attributes according to their degree
            sel_attrs = np.random.choice(attrs_indexes, size=num_a,
                                         replace=False, p=attrs_degrees_prob)

            # Extension of selected attributes
            sel_nodes = extension_csc(sel_attrs, biadjacency_csc)
            sel_g = adjacency[sel_nodes, :][:, sel_nodes].astype(bool) \
                + sparse.identity(len(sel_nodes)).astype(bool)

            # Graph compressor (i.e. description complexity) on induced graph
            mdl = mdl_graph(sel_g)

            # Save result
            if mdl != np.inf and len(sel_nodes) > 0:
                complexity_gen_graphs[len(sel_nodes)].append(mdl)

    return complexity_gen_graphs
