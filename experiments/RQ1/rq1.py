from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutil
import sys

sys.path.append('../..')
from src.algorithm import run_unex_patterns
from src.compressors import generation_complexity
from src.data import load_data
from src.utils import get_root_directory


# ******************************************************** #
# Mine unexpected patterns
# - with constraints on unexpectedness 
# - without constraints 
# ******************************************************** #

DATASETS = ['wikivitals', 'wikivitals-fr', 'wikischools', 'sanFranciscoCrimes', 'ingredients']
S_PARAMS = [8, 7, 6, 5]
DELTA_PARAMS = 0
WITHOUT_CONSTRAINTS = [False, True]
OUTPATH = os.path.join(os.getcwd(), 'output')


def plot_nb_patterns(data_dict: dict, **kwargs):
    """Plot number of patterns according to parameters.

    Parameters
    ----------
    data_dict: dict
        dictionary of values

    Returns
    -------
        Save number of patterns plot.
    """
    markers = ['*', 'o', '+', '^']
    colors = ['#1f78b4', 'firebrick']
    without_constraints = kwargs.get('without_constraints')
    s_params = kwargs.get('S_PARAMS')
    imgpath = kwargs.get('IMGPATH')
    datasets = list(data_dict.keys())

    fig, axes = plt.subplots(1, len(datasets), figsize=(3*len(datasets), 5))

    if len(datasets) == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        for wc in without_constraints:
            if wc:
                ax.plot(data_dict.get(datasets[i]).get(wc), color=colors[1], marker=markers[1],
                        label='Without constraints')
            else:
                ax.plot(data_dict.get(datasets[i]).get(wc), color=colors[0], marker=markers[0],
                        label=r'$\tt{UnexPatterns}$')

        if i == 0:
            ax.set_ylabel('# patterns')
        ax.set_xlabel(r'$s$')
        ax.set_xticks(np.arange(0, len(s_params)), [str(x) for x in s_params])
        ax.set_title(f'{datasets[i]}')
        ax.legend(loc=('upper left'))

    plt.savefig(os.path.join(imgpath, 'rq1.png'), dpi=800)


if __name__ == '__main__':

    # Storage variable for results
    results = defaultdict(dict)

    for dataset in DATASETS:
        print(f"Dataset: {dataset}\n{'='*(len(dataset)+10)}")

        if dataset == 'ingredients':
            BETA_PARAMS = 1
        else:
            BETA_PARAMS = 4
        results[dataset] = defaultdict(list)

        # Load netset data
        adjacency, biadjacency, names, names_col, labels = load_data(dataset)

        # Compute generation complexities
        complexity_gen_graphs = generation_complexity(adjacency, biadjacency, n_attrs=15, n_iter=300)

        for s in S_PARAMS:
            print(f"Parameter s={s}\n{'-'*(len(str(s))+12)}")

            for without_constraints in WITHOUT_CONSTRAINTS:

                # Filename
                if without_constraints:
                    outfilename = f"{dataset}_{str(BETA_PARAMS)}_{str(s)}_{DELTA_PARAMS}_without_constraints"
                else:
                    outfilename = f"{dataset}_{str(BETA_PARAMS)}_{str(s)}_{DELTA_PARAMS}"

                inpath = os.path.join(get_root_directory(), 'output/patterns')
                outfilename_long = 'result_' + outfilename + '.bin'

                # Load patterns if already exist, or mine them otherwise
                if os.path.exists(os.path.join(os.path.join(OUTPATH, 'patterns'), outfilename_long)):
                    print(f"Using existing patterns from {os.path.join(os.path.join(OUTPATH, 'patterns'), outfilename_long)}.")
                    with open(os.path.join(os.path.join(OUTPATH, 'patterns'), outfilename_long), 'rb') as data:
                        patterns = pickle.load(data)
                    nb_patterns = len(patterns)
                elif os.path.exists(os.path.join(inpath, outfilename_long)):
                    print(f'Using existing patterns from {os.path.join(inpath, outfilename_long)}.')
                    shutil.copyfile(os.path.join(inpath, outfilename_long), os.path.join(os.path.join(OUTPATH, 'patterns'), outfilename_long))
                    with open(os.path.join(inpath, outfilename_long), 'rb') as data:
                        patterns = pickle.load(data)
                    nb_patterns = len(patterns)
                else:
                    nb_patterns = run_unex_patterns(adjacency, biadjacency, names_col, complexity_gen_graphs, True, s,
                                                    BETA_PARAMS, DELTA_PARAMS, without_constraints, outfilename,
                                                    OUTPATH)
                
                # Save number of patterns
                results[dataset][without_constraints].append(nb_patterns)

    # Save result
    with open(os.path.join(OUTPATH, 'RQ1.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Plot results
    plot_nb_patterns(results,
                     S_PARAMS=S_PARAMS,
                     without_constraints=WITHOUT_CONSTRAINTS,
                     IMGPATH=os.path.join(OUTPATH, 'img'))


