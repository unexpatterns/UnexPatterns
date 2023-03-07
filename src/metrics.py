import numpy as np
from scipy import sparse


def diversity(pw_distances: np.ndarray, gamma: float = 0.8) -> float:
    """Diversity, i.e. ratio between number of pairwise distances above threshold and total number of distances. 
    
    Parameters
    ----------
    pw_distances: np.ndarray
        Pairwise distances.
    gamma: float (default=0.2)
        Minimumm pairwise distance threshold.
        
    Returns
    -------
        Diversity. 
    """
    n = pw_distances.shape[0]
    if n == 1:
        return 0
    upper = pw_distances[np.triu_indices(n)]
    nb_ps = np.sum(upper > gamma)
    
    return nb_ps / len(upper)


def coverage(summarized_adjacency: sparse.csr_matrix) -> float:
    """Node coverage of summarized graph, i.e. ratio between number of nodes in summarized graph and number of 
    nodes in original graph.
    
    Parameters
    ----------
    summarized_adjacency: sparse.csr_matrix
        Adjacency matrix of the summarized graph
    
    Returns
    -------
        Node coverage. 
    """
    # number of nodes in summarized graph
    n_nodes = len(np.flatnonzero(summarized_adjacency.dot(np.ones(summarized_adjacency.shape[1]))))
    
    # Coverage
    cov = n_nodes / summarized_adjacency.shape[0]
    
    return cov


def coverage_excess(patterns: list, n: int) -> float:
    """Node coverage for Excess algorithm patterns.
    
    Parameters
    ----------
    patterns: list
        List of Excess patterns
    n: int
        Total number of nodes in initial attributed graph. 
        
    Returns
    ------
        Node coverage. 
    """
    all_nodes_excess = set()
    for p in patterns:
        all_nodes_excess |= set(p[0])

    cov = len(all_nodes_excess) / n

    return cov


def width(p_s_labels: np.ndarray, p_s_attributes: np.ndarray) -> float:
    """Width of pattern summaries

    Parameters
    ----------
    p_s_labels: np.ndarray
        Node labels for belonging pattern summaries
    p_s_attributes: np.ndarray
        Pattern summaries x attributes matrix
    
    Returns
    -------
    Float
        Width of pattern summaries
    """
    
    nb_p_s = len(np.unique(p_s_labels))
    nb_nodes_ps = np.mean([np.sum(p_s_labels == i) for i in range(nb_p_s)])
    nb_attrs_ps = np.mean(p_s_attributes.sum(axis=1))
    
    return nb_p_s * np.sqrt(nb_nodes_ps * nb_attrs_ps)


def width_excess(patterns_excess) -> float:
    """Width for Excess patterns.
    
    Parameters
    ----------
    patterns_excess: list
        List of Excess patterns.

    Returns
    -------
        Width. 
    """
    nb_nodes, nb_attrs = [], []
    for i in patterns_excess:
        nb_nodes.append(len(i[0]))
        nb_attrs.append(len(i[1]))

    nb_e_patterns = len(patterns_excess)

    return nb_e_patterns * np.sqrt(np.mean(nb_nodes) * np.mean(nb_attrs))


def expressiveness(summarized_adjacency: sparse.csr_matrix, pw_distances: np.ndarray, gamma: float, 
                   p_s_labels: np.ndarray, p_s_attributes: np.ndarray):
    """Expressiveness of pattern summaries, i.e. (diversity x corevage) / width.
    
    Parameters
    ----------
    summarized_adjacency: sparse.csr_matrix
        Summarized adjacency of the graph
    pw_distances: np.ndarray
        Pairwise distances
    gamma: float
        Minimumm pairwise distance threshold
    p_s_labels: np.ndarray
        Node labels for belonging pattern summaries
    p_s_attributes: np.ndarray
        Pattern summaries x attributes matrix

    Returns
    -------
        Pattern summaries expressiveness.
    """
    div = diversity(pw_distances, gamma)
    cov = coverage(summarized_adjacency)
    wid = width(p_s_labels, p_s_attributes)
    expr = (div * cov) / wid

    return div, cov, wid, expr
