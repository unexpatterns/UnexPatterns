"""Sias-Miner"""
from collections import defaultdict
import json
import os
import sys
import numpy as np

sys.path.append('../..')
from src.data import get_pw_distance_matrix, load_data, preprocess_data, read_parameters, get_excess_pattern, get_sias_pattern
from src.distances import pairwise_wd_distance
from src.metrics import coverage_excess, diversity, width_excess
from src.utils import get_gensim_model, get_root_directory


if __name__ == '__main__':

    # Get parameters
    PARAM_FILENAME = sys.argv[1]
    parameters = read_parameters(PARAM_FILENAME)
    INPATH = os.path.join(os.getcwd(), parameters.get('patterns_path'))
    MODEL_PATH = os.path.join(get_root_directory(), 'models')

    expr_sias = defaultdict(list)

    for dataset in parameters.get('datasets'):
        print(f'****Dataset: {dataset}')

        # Load netset data
        adjacency, biadjacency, names, names_col, labels = load_data(dataset)
        # Gensim model
        model = get_gensim_model(MODEL_PATH, f'gensim_model_{dataset}', biadjacency, names_col)

        for s_param in parameters.get('s'):
            print(f'--s={s_param}')

            # Preprocess data (get same attribute order as in UnexPattern)
            new_biadjacency, words = preprocess_data(biadjacency, names_col,
                                                     s_param, sort_data=False)

            # Load Sias patterns
            with open(f'{INPATH}/{dataset}/retrievedPatterns_{s_param}.json', 'rb') as f:
                data = json.load(f)

            nb_sias_patterns = len(data.get('patterns'))

            # Preprocess Sias patterns
            sias_patterns = [get_sias_pattern(data.get('patterns')[idx], names, names_col) for idx in range(nb_sias_patterns)]
            sias_patterns_filt = [p for p in sias_patterns if len(p[0]) >= s_param]

            # Pattern x attributes matrix
            nb_sias_patterns_filt = len(sias_patterns_filt)
            if dataset != 'sanFranciscoCrimes':
                sias_patterns_attributes = np.zeros((nb_sias_patterns_filt, new_biadjacency.shape[1]))
            else:
                sias_patterns_attributes = np.zeros((nb_sias_patterns_filt, biadjacency.shape[1]))
            for i, p in enumerate(sias_patterns_filt):
                sias_patterns_attributes[i, p[1]] = 1

            # Wasserstein distances
            wd_filename = f'wasserstein_distances_{dataset}_5_{s_param}_sias_patterns.pkl'
            if os.path.exists(f'{INPATH}/{dataset}/{wd_filename}'):
                wd_distances_sias = get_pw_distance_matrix(dataset, beta=5, s=s_param,
                                                           path=os.path.join(INPATH, dataset),
                                                           method='sias_patterns')
            else:
                wd_distances_sias = pairwise_wd_distance(sias_patterns_attributes, nb_sias_patterns_filt, model,
                                                         names_col)
                # Save distances
                with open(f'{INPATH}/{dataset}/{wd_filename}', 'wb') as f:
                    np.save(f, wd_distances_sias)

            # Expressiveness metric
            div = diversity(wd_distances_sias, gamma=parameters.get('gamma'))
            cov = coverage_excess(sias_patterns_filt, adjacency.shape[0])
            width = width_excess(sias_patterns_filt)
            expr = (div * cov) / width
            expr_sias[dataset].append(expr)

            with open(f"{INPATH}/{dataset}/expressiveness_{dataset}_5_{s_param}_{parameters.get('gamma')}_sias.txt", 'w') as f:
                f.write(f'{div}, {cov}, {width}, {expr}')
