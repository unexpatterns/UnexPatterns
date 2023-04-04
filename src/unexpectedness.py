"""Unexpectedness"""
from typing import Union
import numpy as np
from scipy import sparse, special

from src.compressors import mdl_graph


def graph_unexpectedness(adjacency: sparse.csr_matrix, gen_complexities: dict) -> float:
    """Unexpectedness of a graph structure.

    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph.
    gen_complexities: dict
         Dictionary with number of nodes as keys and list of graph generation complexities as values.

    Returns
    -------
        Unexpectedness of a graph structure as a float value. """
    n = adjacency.shape[0]
    complexity_desc_g = mdl_graph(adjacency.astype(bool) + sparse.identity(n).astype(bool))
    try:
        avg = np.mean(gen_complexities.get(n))
    except TypeError:
        avg = 0

    complexity_gen_g = avg

    return complexity_gen_g - complexity_desc_g


def attr_unexpectedness(attribute: int, extent_prev: list, extent: list,
                        degrees: np.ndarray) -> float:
    """Unexpectedness of attribute.

    Parameters
    ----------
    attribute: int
        Index of attribute.
    extent_prev: list
        Extension of pattern attributes at previous iteration.
    extent: list
        Current pattern extension.
    degrees: np.ndarray
        Array of attribute degrees in biadjacency.

    Returns
    -------
        Pattern attribute unexpectedness. """
    cw_a = np.log2(1 / (degrees[attribute] / np.sum(degrees)))
    cd_a = np.log2((1 / (len(extent) / len(extent_prev))) + 1)

    return cw_a - cd_a


def pattern_unexpectedness(adjacency: sparse.csr_matrix,
                           biadjacency: sparse.csr_matrix,
                           gen_complexities: dict,
                           attributes: list, degrees: np.ndarray) -> float:
    """Pattern unexpectedness, as the sum of the unexpectedness of its elements.

    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph.
    biadjacency: sparse.csr_matrix
        Features matrix of the graph. Contains nodes x attributes.
    gen_complexities: dict
        Dictionnary with number of nodes as keys and list of graph generation
        complexities as values.
    attributes: list
        List of attribute indexes.
    degrees: np.ndarray
        Array of attribute degrees in biadajcency

    Returns
    -------
        Unexpectedness of pattern as a float value. """

    u_g = graph_unexpectedness(adjacency, gen_complexities)
    u_a = attr_unexpectedness(biadjacency, attributes, degrees)

    return u_g + u_a
