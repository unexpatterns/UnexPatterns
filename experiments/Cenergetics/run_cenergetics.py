"""Cenergetics"""
from collections import defaultdict
import json
import os
import sys
import numpy as np

sys.path.append('../..')
from src.data import get_pw_distance_matrix, load_data, preprocess_data, read_parameters, get_excess_pattern, \
    get_sias_pattern
from src.distances import pairwise_wd_distance
from src.metrics import coverage_excess, diversity, width_excess
from src.utils import get_gensim_model, get_root_directory


if __name__ == '__main__':

    # Get parameters
    PARAM_FILENAME = sys.argv[1]
    parameters = read_parameters(PARAM_FILENAME)
    INPATH = os.path.join(os.getcwd(), parameters.get('patterns_path'))
    MODEL_PATH = os.path.join(get_root_directory(), 'models')

    expr_cenergetics = defaultdict(list)

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

            # Load Cenergetics patterns
            with open(f'{INPATH}/{dataset}/retrievedPatterns_{s_param}.json', 'rb') as f:
                data = json.load(f)

            nb_cenergetics_patterns = len(data.get('patterns'))

            # Preprocess Cenergetics patterns
            if dataset != 'sanFranciscoCrimes':
                cenergetics_patterns = [get_sias_pattern(data.get('patterns')[idx]) for idx in range(nb_cenergetics_patterns)]
                cenergetics_patterns_filt = [p for p in cenergetics_patterns if len(p[0]) >= s_param]
            else:
                cenergetics_patterns = [get_excess_pattern(data.get('patterns')[idx], names, names_col) for idx in range(nb_cenergetics_patterns)]
                cenergetics_patterns_filt = [p for p in cenergetics_patterns if len(p[0]) >= s_param]

            # Pattern x attributes matrix
            nb_cenergetics_patterns_filt = len(cenergetics_patterns_filt)
            if dataset != 'sanFranciscoCrimes':
                cenergetics_patterns_attributes = np.zeros((nb_cenergetics_patterns_filt, new_biadjacency.shape[1]))
            else:
                cenergetics_patterns_attributes = np.zeros((nb_cenergetics_patterns_filt, biadjacency.shape[1]))
            for i, p in enumerate(cenergetics_patterns_filt):
                cenergetics_patterns_attributes[i, p[1]] = 1

            # Wasserstein distances
            wd_filename = f'wasserstein_distances_{dataset}_5_{s_param}_cenergetics_patterns.pkl'
            if os.path.exists(f'{INPATH}/{dataset}/{wd_filename}'):
                wd_distances_cenergetics = get_pw_distance_matrix(dataset, beta=5, s=s_param,
                                                                  path=os.path.join(INPATH, dataset),
                                                                  method='cenergetics_patterns')
            else:
                wd_distances_cenergetics = pairwise_wd_distance(cenergetics_patterns_attributes,
                                                                nb_cenergetics_patterns_filt, model, names_col)
                # Save distances
                with open(f'{INPATH}/{dataset}/{wd_filename}', 'wb') as f:
                    np.save(f, wd_distances_cenergetics)

            # Expressiveness metric
            div = diversity(wd_distances_cenergetics, gamma=parameters.get('gamma'))
            cov = coverage_excess(cenergetics_patterns_filt, adjacency.shape[0])
            width = width_excess(cenergetics_patterns_filt)
            expr = (div * cov) / width
            expr_cenergetics[dataset].append(expr)

            with open(f"{INPATH}/{dataset}/expressiveness_{dataset}_5_{s_param}_{parameters.get('gamma')}_cenergetics.txt", 'w') as f:
                f.write(f'{div}, {cov}, {width}, {expr}')
