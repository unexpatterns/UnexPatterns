import numpy as np
from scipy import sparse

from sknetwork.clustering import Louvain, KMeans
from sknetwork.gnn import GNNClassifier
from sknetwork.utils import KMeansDense


def get_louvain(dataset: str, adjacency: sparse.csr_matrix, nb_cc: int, resolutions: dict) -> np.ndarray:
    """Louvain algorithm for clustering graphs by maximization of modularity. Return labels of the nodes.
    
    Parameters
    ----------
    dataset: str
        Name of dataset on netset.
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    nb_cc: int
        Number of communities
    resolutions: dict
        Resolution values for each value of nb_cc
    
    Returns
    -------
        Array of node labels.
    """
    louvain = Louvain(resolution=resolutions.get(dataset).get(nb_cc)) 
    labels_louvain = louvain.fit_transform(adjacency)

    return labels_louvain


def get_gnn(adjacency: sparse.csr_matrix, biadjacency: sparse.csr_matrix, labels: np.ndarray, hidden_dim: int, 
            nb_cc: int) -> np.ndarray:
    """GNN embedding + KMeans clustering. Return labels of the nodes.
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    biadjacency: sparse.csr_matrix
        Biadjacency matrix of the graph
    labels: np.ndarray
        Node labels
    hidden_dim: int
        Hidden layer dimension
    nb_cc: int
        Number of communities (for KMeans clustering)
    
    Returns
    -------
        Array of node labels.
    """
    features = biadjacency
    n_labels = len(np.unique(labels))
    gnn = GNNClassifier(dims=[hidden_dim, n_labels],
                        layer_types='conv',
                        activations=['Relu', 'Softmax'],
                        verbose=False)

    # Train GNN model
    gnn.fit(adjacency, features, labels, train_size=0.8, val_size=0.1, test_size=0.1, n_epochs=50)
    
    # KMeans on GNN node embedding
    gnn_embedding = gnn.layers[-1].embedding
    kmeans = KMeansDense(n_clusters=nb_cc)  # k = number of connected components in summarized graph
    kmeans_gnn_labels = kmeans.fit_transform(gnn_embedding)

    return kmeans_gnn_labels


def get_spectral(adjacency: sparse.csr_matrix,  nb_cc: int) -> np.ndarray:
    """Spectral embedding + KMeans clustering. Return labels of the nodes.
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    nb_cc: int
        Number of communities (for KMeans clustering)
    
    Returns
    -------
        Array of node labels.
    """
    # Spectral + KMeans
    kmeans = KMeans(n_clusters=nb_cc)  # k = number of connected components in summarized graph
    kmeans_spectral_labels = kmeans.fit_transform(adjacency)

    return kmeans_spectral_labels


def get_doc2vec(model, nb_cc: int) -> np.ndarray:
    """Doc2Vec embedding + KMeans clustering. Return labels of the nodes.
    
    Parameters
    ----------
    model:
        Pre-trained gensim model.
    nb_cc: int
        Number of communities (for KMeans clustering)
    
    Returns
    -------
        Array of node labels.
    """
    # Kmeans on embeddings
    kmeans = KMeansDense(n_clusters=nb_cc)  # k = number of connected components in summarized graph
    kmeans_doc2vec_labels = kmeans.fit_transform(model.dv.vectors)

    return kmeans_doc2vec_labels


def get_community_graph(adjacency: sparse.csr_matrix, labels_communities: np.ndarray) -> sparse.csr_matrix:
    """Equivalent of summarized graph but for community-based methods. Return the adjacency matrix of the graph
    made of the union of all communities. 
    
    Parameters
    ----------
    adjacency: sparse.csr_matrix
        Adjacency matrix of the graph
    labels_communities: np.ndarray
        Array of node community labels 

    Returns
    ------
        Sparse matrix of the community graph.
    """
    n_com = len(np.unique(labels_communities))
    rows, cols = [], []
    for com in range(n_com):
        nodes = np.flatnonzero(labels_communities == com)
        idx = 0
        idx_nodes = np.array([-1] * len(nodes))  # number of unique nodes from communities
        # reindex nodes
        for n in nodes:
            if n not in idx_nodes:
                idx_nodes[idx] = n
                idx += 1

        # Record edges from subgraph related to community
        adj_com = adjacency[nodes, :][:, nodes].tocoo()
        reindex_rows = [int(idx_nodes[src]) for src in adj_com.row]
        reindex_cols = [int(idx_nodes[dst]) for dst in adj_com.col]
        rows += reindex_rows
        cols += reindex_cols
        
    return sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=adjacency.shape).tocsr()
