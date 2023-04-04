"""Algorithms"""
import pickle
from contextlib import redirect_stdout
from typing import List, Optional
from line_profiler import LineProfiler
import numpy as np
from scipy import stats, sparse

from sknetwork.utils import get_degrees

from src.data import preprocess_data
from src.derivation import extension, intention
from src.unexpectedness import graph_unexpectedness, attr_unexpectedness
from src.utils import smoothing, shuffle_columns


def is_cannonical(context: sparse.csr_matrix, extents: list, intents: list,
                  r: int, y: int):
    """Verify if an extent has already been seen (part of InClose original algorithm)."""

    global r_new

    for k in range(len(intents[r]) - 1, -1, -1):
        for j in range(y, intents[r][k], -1):
            for h in range(len(extents[r_new])):
                if context[extents[r_new][h], j] == 0:
                    h -= 1  # Necessary for next test in case last interaction of h for-loop returns False
                    break
            if h == len(extents[r_new]) - 1:
                return False
        y = intents[r][k] - 1

    for j in reversed(range(y, -1, -1)):
        for h in range(len(extents[r_new])):
            if context[extents[r_new][h], j] == 0:
                h -= 1  # Necessary for next test in case last interaction of h for-loop returns False
                break
        if h == len(extents[r_new]) - 1:
            return False

    return True


def init_unex_patterns(context: sparse.csr_matrix) -> tuple:
    """Initialization for UnexPatterns algorithm.

    Parameters
    ---------
    context: sparse.csr_matrix
        Features matrix of the graph. Contains nodes x attributes.

    Returns
    -------
        Tuple of two lists, containing all nodes in graph and empty list of attributes. """
    extents, intents = [], []
    extents_init = np.arange(context.shape[0])
    intents_init = []
    extents.append(extents_init)  # Initalize extents with all objects from context
    intents.append(intents_init)  # Initialize intents with empty set attributes

    unexs = [0]

    return extents, intents, unexs


def unex_patterns(adjacency: sparse.csr_matrix,
                  context: sparse.csr_matrix,
                  context_csc: sparse.csr_matrix,
                  extents: list,
                  intents: list,
                  r: int = 0,
                  y: int = 0,
                  min_support: int = 0,
                  max_support: int = np.inf,
                  beta: int = 0,
                  delta: int = 0,
                  degs: Optional[list] = [],
                  unexs_g: Optional[list] = [],
                  unexs_a: Optional[list] = [],
                  unexs: Optional[list] = [],
                  names_col: Optional[list] = [],
                  comp_gen_graph: Optional[dict] = None,
                  without_constraints: bool = False,
                  shuf: bool = False) -> List:
    """Mining pattern algorithm using Unexpectedness + IsCannonical function (derived from InClose algorithm).

    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    context: sparse.csr_matrix
        Features matrix of the graph. Contains nodes x attributes.
    context_csc: sparse.csc_matrix
        Features matrix of the graph in CSC format.
    extents: list
        List of extents, i.e. sets of nodes.
    intents: list
        List of intents, i.e. sets of attributes.
    r: int (default=0)
        Index of the pattern being filled.
    y: int (default=0)
        Index of candidate attribute.
    min_support: int (default=0)
        Minimum support value for extent.
    max_support: int (default +inf)
        Maximum support value for extent.
    beta: int (default=0)
        Minimum support value for intent.
    delta: int (default=0)
        Minimum value for Unexpectedness difference between patterns.
    degs, unexs_g, unexs_a, unexs, names_col: list
        Lists for value storage over recursion.
    comp_gen_graph: dict (default=None)
        Dictionary with number of nodes as keys and list of graph generation complexities as values.
    without_constraints: bool (default=False)
        If True, remove unexpectedness constraint as well as probability of reordering attributes.

    Returns
    -------
        List of tuples where each tuple is an unexpected pattern made of (extent, intent).
    """
    global r_new
    global ptr
    r_new = r_new + 1

    for j in np.arange(context.shape[1])[y:]:
        try:
            extents[r_new] = []
            unexs_g[r_new] = 0
            unexs_a[r_new] = 0
        except IndexError:
            extents.append([])
            unexs_g.append(0)
            unexs_a.append(0)

        # Form a new extent by adding extension of attribute j to current pattern extent
        ext_j = set(extension([j], context_csc))
        extents[r_new] = list(sorted(set(extents[r]).intersection(ext_j)))
        len_new_extent = len(extents[r_new])

        if (len_new_extent >= min_support) and (len_new_extent <= max_support):

            # Verify that length of intention of new extent is greater than a threshold (e.g beta)
            # In other words, we only enter the loop if the new extent still has "space" to welcome enough new
            # attributes.
            # Using this, we can trim all patterns with not enough attributes from the recursion tree
            size_intention = len(intention(extents[r_new], context))

            if size_intention >= beta:

                new_intent = list(sorted(set(intents[r]).union({j})))

                # Compute Unexpectedness on pattern
                unex_g = graph_unexpectedness(adjacency[extents[r_new], :][:, extents[r_new]], comp_gen_graph)
                unexs_g[r_new] = unex_g
                # Attributes unexpectedness
                unex_a = attr_unexpectedness(j, extents[r], extents[r_new], degs)
                unex_a += unexs_a[r]
                unexs_a[r_new] = unex_a
                # Total unexpectedness
                unex = unex_g + unex_a

                if len_new_extent - len(extents[r]) == 0:

                    if without_constraints or unex - unexs[ptr] >= delta:
                        intents[r] = new_intent
                        unexs[-1] = unex
                    else:
                        break
                else:
                    is_canno = is_cannonical(context, extents, intents, r, j - 1)
                    if is_canno:
                        try:
                            intents[r_new] = []
                        except IndexError:
                            intents.append([])

                        if unex - unexs[ptr] >= delta or r == 0:
                            intents[r_new] = new_intent
                            unexs.append(unex)
                            ptr += 1

                            # Probability of reordering attribute list
                            p = smoothing(len(unexs))
                            X = stats.bernoulli(p)

                            if not shuf and X.rvs(1)[0] == 1:
                                start = j + 1
                                end = len(names_col)
                                rand_idxs = np.random.choice(np.arange(start, end), size=(end - start), replace=False)
                                if len(rand_idxs) > 0:
                                    context = shuffle_columns(context, rand_idxs)
                                    context_csc = context.tocsc()
                                    new_names_col = shuffle_columns(names_col, rand_idxs)
                                    degs = shuffle_columns(degs, rand_idxs)
                                    names_col = new_names_col
                                    shuf = True
                            elif shuf:
                                sort_index = np.argsort(get_degrees(context.astype(bool), transpose=True))
                                new_degs = degs[sort_index]
                                new_context = context[:, sort_index]
                                new_names_col = names_col[sort_index]
                                context_csc = new_context.tocsc()
                                degs = new_degs.copy()
                                context = new_context.copy()
                                names_col = new_names_col.copy()
                                shuf = False

                            unex_patterns(adjacency, context, context_csc, extents, intents, r=r_new, y=j + 1,
                                          min_support=min_support, max_support=max_support, beta=beta, delta=delta,
                                          degs=degs, unexs_g=unexs_g, unexs_a=unexs_a, unexs=unexs,
                                          names_col=names_col, comp_gen_graph=comp_gen_graph,
                                          without_constraints=without_constraints, shuf=shuf)
                        else:
                            break

    unexs.pop(-1)
    ptr -= 1
    shuf = False

    return [*zip(extents, intents)]


def run_unex_patterns(adjacency: sparse.csr_matrix,
                      biadjacency: sparse.csr_matrix,
                      names_col: np.ndarray,
                      complexity_gen_graphs: dict,
                      order_attributes: bool,
                      s: int,
                      beta: int,
                      delta: int,
                      without_constraints: bool,
                      outfile: str,
                      outpath: str):
    """Run pattern mining algorithm.

    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph.
    biadjacency: sparse.csr_matrix
        Features matrix of the graph. Contains nodes x attributes.
    names_col: np.ndarray
        Features names.
    complexity_gen_graphs: dict
        Dictionary with number of nodes as keys and list of graph generation complexities as values.
    order_attributes: bool
        If True, order attributes according to their ascending degree.
    s: int
        Minimum extent support.
    beta: int
        Minimum intent support.
    delta: int
        Minimum value for Unexpectedness difference between patterns.
    without_constraints: bool
        If True, remove unexpectedness constraint as well as probability of reordering attributes.
    outfile: str
        Output filename.
    outpath: str
        Output path name.
    """
    # Initialization
    extents, intents, unexs = init_unex_patterns(biadjacency)
    global r_new
    r_new = 0
    global ptr
    ptr = 0

    # Preprocess data according to parameters
    new_biadjacency, words, freq_attributes = preprocess_data(biadjacency, names_col, s, order_attributes,
                                                              return_degs=True)

    # Convert context to csc at first, to fasten algorithm
    new_biadjacency_csc = new_biadjacency.tocsc()
    print(f'Context: {new_biadjacency.shape}')

    # Algorithm
    print('Mining patterns...')
    with open(f'{outpath}/logs/log_{outfile}', 'w') as f:
        with redirect_stdout(f):
            print('starts profiling...')
            lp = LineProfiler()
            lp_wrapper = lp(unex_patterns)
            lp_wrapper(adjacency, new_biadjacency, new_biadjacency_csc, extents, intents, r=0, y=0, min_support=s,
                       max_support=np.inf, beta=beta, delta=delta, degs=freq_attributes, unexs_g=[0], unexs_a=[0],
                       unexs=unexs, names_col=words, comp_gen_graph=complexity_gen_graphs,
                       without_constraints=without_constraints, shuf=False)
            lp.print_stats()

    res = [*zip(extents, intents)]
    print(f'Found {len(res)} patterns!')

    # Save result
    print(f'Saving results in {outpath}...')
    with open(f"{outpath}/patterns/result_{outfile}.bin", "wb") as output:
        pickle.dump(res, output)

    return len(res)
