import json
import numpy as np
from tqdm import tqdm

from src.data import load_data

from sknetwork.utils import get_degrees


def dataset2json(dataset: str, outpath: str, min_support: int = 0, max_support: int = np.inf):
    """Save sknetwork dataset to SIAS-Miner JSON format.
    
    Parameters
    ----------
    dataset: str
        Dataset name
    outpath: str
        Output path
    min_support: int
        Minimum support for number of attributes
    max_support: int
        Maximum support for number of attributes
    """
        
    adjacency, biadjacency, names, names_col, _ = load_data(dataset)
    n = adjacency.shape[0] 
    out_graph = {}
    print(f'Number of attributes: {biadjacency.shape[1]}')

    # Degree of attribute = # articles in which it appears
    freq_attribute = get_degrees(biadjacency.astype(bool), transpose=True)
    index = np.flatnonzero((freq_attribute <= max_support) & (freq_attribute >= min_support))

    # Filter data with index
    biadjacency = biadjacency[:, index]
    m = biadjacency.shape[1]
    print(f'new number of attributes: {m}')

    # Dataset name
    out_graph['descriptorName'] = dataset

    # Attribute names
    out_graph['attributesName'] = [str(i) for i in range(m)]

    # Vertices: nodes + attributes
    vertices = []
    for u in tqdm(range(n)):
        tmp = {'vertexId': str(u)}
        feats = np.zeros(m)
        feats[biadjacency[u].indices] = biadjacency[u].data
        tmp['descriptorsValues'] = list(map(int, feats))
        vertices.append(tmp)
    out_graph['vertices'] = vertices
    print(f'Verttices done!')

    # Edges
    edges = [{'vertexId': str(u), 'connected_vertices': list(adjacency[u].indices.astype(str))} for u in range(n)]
    out_graph['edges'] = edges
    print(f'Edges done!')

    # Save data
    print(f'Saving data...')
    with open(outpath, 'w') as f:
        json.dump(out_graph, f)


# ==================================================================
MIN_SUPPORT = 8
MAX_SUPPORT = 1000
DATASETS = ['wikivitals', 'wikivitals-fr', 'wikischools']


if __name__ == '__main__':

    for dataset in DATASETS:
        dataset2json(dataset, f'data/graph_{dataset}_max_support_{MAX_SUPPORT}_min_support_{MIN_SUPPORT}.json',
                     min_support=MIN_SUPPORT, max_support=MAX_SUPPORT)
        print(f'{dataset} done!')
