from collections import defaultdict
import numpy as np
import os
import pickle
import shutil
import sys

sys.path.append('../..')
from src.algorithm import run_unex_patterns
from src.baselines import get_community_graph, get_doc2vec, get_gnn, get_louvain, get_spectral
from src.compressors import generation_complexity
from src.data import get_pw_distance_matrix, load_data, load_patterns, preprocess_data
from src.distances import pairwise_wd_distance
from src.metrics import expressiveness
from src.summarization import get_pattern_summaries, get_summarized_biadjacency, get_summarized_graph
from src.utils import get_gensim_model, get_root_directory, p_s_attributes, save_matrix


# ******************************************************** #
# Mine unexpected patterns and:
# - form pattern summaries
# - compute distances between pattern summaries
# - compute expressiveness of pattern summaries
# - compare to baselines approaches
# ******************************************************** #

DATASETS = ['sanFranciscoCrimes', 'ingredients', 'wikischools', 'wikivitals-fr', 'wikischools']
S_PARAMS = [8, 7, 6, 5]
DELTA_PARAMS = 0
OUTPATH = os.path.join(os.getcwd(), 'output')
MODEL_PATH = os.path.join(get_root_directory(), 'models')
METHODS = ['summaries', 'louvain', 'gnn_kmeans', 'spectral_kmeans', 'd2v_kmeans']

# Louvain resolutions crafted for each number of patterns found (for each configuration of dataset and parameters). 
# If necessary, fill this dictionary with additional values.
with open(os.path.join(OUTPATH, 'louvain_resolutions.pkl'), 'rb') as f:
    RESOLUTIONS = pickle.load(f)


if __name__ == '__main__':

    # Storage variable for results
    results = defaultdict(dict)

    for dataset in DATASETS:
        print(f"Dataset: {dataset}\n{'='*(len(dataset)+10)}")

        if dataset == 'ingredients':
            BETA_PARAMS = 1
            GAMMA_PARAMS = 0.05
        else:
            BETA_PARAMS = 4
            if dataset == 'sanFranciscoCrimes':
                GAMMA_PARAMS = 0.2
            else:
                GAMMA_PARAMS = 0.8
        
        results[dataset] = defaultdict(list)

        # Load netset data
        adjacency, biadjacency, names, names_col, labels = load_data(dataset)

        # Gensim model
        model = get_gensim_model(MODEL_PATH, f'gensim_model_{dataset}', biadjacency, names_col)
        print(f'Model information: {model.wv}')

        # Compute generation complexities
        complexity_gen_graphs = generation_complexity(adjacency, biadjacency, n_attrs=15, n_iter=300)

        for s in S_PARAMS:
            print(f"Parameter s={s}\n{'-'*(len(str(s))+12)}")

            # Preprocess data (get same attribute order as in UnexPattern)
            new_biadjacency, words = preprocess_data(biadjacency, names_col, s, sort_data=True)

            # Filename
            inpath = os.path.join(get_root_directory(), 'output/patterns')
            outfilename = f"{dataset}_{str(BETA_PARAMS)}_{str(s)}_{DELTA_PARAMS}"
            outfilename_long = 'result_' + outfilename + '.bin'

            # Load patterns if already exist, or mine them otherwise
            if os.path.exists(os.path.join(os.path.join(OUTPATH, 'patterns'), outfilename_long)):
                print(f"Using existing patterns from {os.path.join(OUTPATH, 'patterns')}.")
                patterns = load_patterns(dataset=dataset, 
                                         beta=BETA_PARAMS, 
                                         s=s, 
                                         order=True, 
                                         inpath=os.path.join(OUTPATH, 'patterns'), 
                                         with_prob=True, 
                                         delta=DELTA_PARAMS)

            elif os.path.exists(os.path.join(inpath, outfilename_long)):
                print(f'Using existing patterns from {os.path.join(inpath, outfilename_long)}.')
                shutil.copyfile(os.path.join(inpath, outfilename_long), os.path.join(os.path.join(OUTPATH, 'patterns'), 
                                                                                     outfilename_long))
                patterns = load_patterns(dataset=dataset, 
                                         beta=BETA_PARAMS, 
                                         s=s, 
                                         order=True, 
                                         inpath=inpath, 
                                         with_prob=True, 
                                         delta=DELTA_PARAMS)

            else:
                nb_patterns = run_unex_patterns(adjacency, 
                                                biadjacency, 
                                                names_col, 
                                                complexity_gen_graphs, 
                                                True, 
                                                s, 
                                                BETA_PARAMS, 
                                                DELTA_PARAMS, 
                                                False, 
                                                outfilename, 
                                                OUTPATH)
                patterns = load_patterns(dataset=dataset,
                                         beta=BETA_PARAMS,
                                         s=s,
                                         order=True,
                                         inpath=inpath,
                                         with_prob=True,
                                         delta=DELTA_PARAMS)
                
            # Graph summarization
            summarized_adjacency = get_summarized_graph(adjacency, patterns)            
            summarized_biadjacency = get_summarized_biadjacency(adjacency, new_biadjacency, patterns)

            # Pattern summaries
            labels_cc_summarized, mask = get_pattern_summaries(summarized_adjacency)
            nb_p_s = len(np.unique(labels_cc_summarized))
            p_s_attributes_summaries = p_s_attributes(summarized_biadjacency, labels_cc_summarized, mask)

            # Baselines patterns/communities
            # --------------------------------------------------------------------
            print('Computing patterns for baselines...')

            # Louvain
            if nb_p_s > 1 and dataset != 'ingredients':
                labels_louvain = get_louvain(dataset, adjacency, nb_p_s, RESOLUTIONS)
                nb_louvain = len(np.unique(labels_louvain))
                p_s_attributes_louvain = p_s_attributes(biadjacency, labels_louvain)
                louvain_adjacency = get_community_graph(adjacency, labels_louvain)

            # GNN
            if len(labels) > 0:
                kmeans_gnn_labels = get_gnn(adjacency, biadjacency, labels, 
                                            hidden_dim=16, nb_cc=nb_p_s)
                p_s_attributes_gnn_kmeans = p_s_attributes(biadjacency, kmeans_gnn_labels)
                gnn_adjacency = get_community_graph(adjacency, kmeans_gnn_labels)

            # Spectral + KMeans
            kmeans_spectral_labels = get_spectral(adjacency, nb_cc=nb_p_s)
            p_s_attributes_spectral = p_s_attributes(biadjacency, kmeans_spectral_labels)
            kmeans_spectral_adjacency = get_community_graph(adjacency, kmeans_spectral_labels)

            # KMeans model on d2v embeddings
            kmeans_doc2vec_labels = get_doc2vec(model, nb_cc=nb_p_s)
            p_s_attributes_doc2vec_kmeans = p_s_attributes(biadjacency, kmeans_doc2vec_labels)
            kmeans_doc2vec_adjacency = get_community_graph(adjacency, kmeans_doc2vec_labels)

            # Pattern summaries pairwise distances
            # --------------------------------------------------------------------
            pw_distances_dict = {}
            dist_path_dir = os.path.join(OUTPATH, 'distances')

            for i, method in enumerate(METHODS):
                dist_filename = f'wasserstein_distances_{dataset}_{BETA_PARAMS}_{s}_{method}.pkl'

                if os.path.exists(os.path.join(dist_path_dir, dist_filename)):
                    print(f"Use existing file in {os.path.join(dist_path_dir, dist_filename)}")
                    pw_distances = get_pw_distance_matrix(dataset, BETA_PARAMS, s, dist_path_dir, method=method)
                    
                else:
                    distance_path = os.path.join(dist_path_dir, dist_filename)
                    if method == 'summaries':
                        pw_distances = pairwise_wd_distance(p_s_attributes_summaries, nb_p_s, model, words)
                        save_matrix(pw_distances, distance_path)
                    else:
                        if method == 'louvain' and nb_p_s > 1 and dataset != 'ingredients':
                            pw_distances = pairwise_wd_distance(p_s_attributes_louvain, nb_louvain, model, names_col)
                        elif method == 'gnn_kmeans' and len(labels) > 0:
                            pw_distances = pairwise_wd_distance(p_s_attributes_gnn_kmeans, nb_p_s, model, names_col)
                        elif method == 'spectral_kmeans':
                            pw_distances = pairwise_wd_distance(p_s_attributes_spectral, nb_p_s, model, names_col)
                        elif method == 'd2v_kmeans':
                            pw_distances = pairwise_wd_distance(p_s_attributes_doc2vec_kmeans, nb_p_s, model, names_col)
                        save_matrix(pw_distances, distance_path)
                
                pw_distances_dict[method] = pw_distances

            # Expressiveness of results
            # --------------------------------------------------------------------
            expr_path = os.path.join(OUTPATH, 'expressiveness')

            for method in METHODS:
                filename = f'expressiveness_details_{dataset}_{BETA_PARAMS}_{s}_{method}_{GAMMA_PARAMS}.txt'

                if method == 'summaries':
                    div, cov, wid, expr = expressiveness(summarized_adjacency, pw_distances_dict.get(method),
                                                         GAMMA_PARAMS,
                                                         labels_cc_summarized,
                                                         p_s_attributes_summaries)
                elif method == 'louvain' and nb_p_s > 1 and dataset != 'ingredients':
                    div, cov, wid, expr = expressiveness(louvain_adjacency, 
                                                         pw_distances_dict.get(method),
                                                         GAMMA_PARAMS,
                                                         labels_louvain,
                                                         p_s_attributes_louvain)
                elif method == 'gnn_kmeans' and len(labels) > 0:
                    div, cov, wid, expr = expressiveness(gnn_adjacency, 
                                                         pw_distances_dict.get(method),
                                                         GAMMA_PARAMS,
                                                         kmeans_gnn_labels,
                                                         p_s_attributes_gnn_kmeans)
                elif method == 'spectral_kmeans':
                    div, cov, wid, expr = expressiveness(kmeans_spectral_adjacency, 
                                                         pw_distances_dict.get(method),
                                                         GAMMA_PARAMS,
                                                         kmeans_spectral_labels,
                                                         p_s_attributes_spectral)
                elif method == 'd2v_kmeans':
                    div, cov, wid, expr = expressiveness(kmeans_doc2vec_adjacency, 
                                                         pw_distances_dict.get(method),
                                                         GAMMA_PARAMS,
                                                         kmeans_doc2vec_labels,
                                                         p_s_attributes_doc2vec_kmeans)

                # Save result
                with open(os.path.join(expr_path, filename), 'w') as f:
                    f.write(f'{div}, {cov}, {wid}, {expr}')
