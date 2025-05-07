import numpy as np
import xgi
import pandas as pd
from sklearn import metrics
import networkx as nx
from sklearn import preprocessing, metrics
import random
import community as community_louvain
from networkx.algorithms import community
from networkx.algorithms.community import girvan_newman, label_propagation_communities
from infomap import Infomap
import igraph as ig
from sklearn.metrics import normalized_mutual_info_score

def main():
    global H, _weight, Y, edges

    H = xgi.Hypergraph()  # Create an empty hypergraph object
    # Parameter settings
    n = 100  # Number of nodes in the hypergraph
    avg_degree = 15  # Average degree of each node
    num_communities = 3  # Number of communities in the hypergraph
    intra_strength = 0.85  # Strength of connections within communities


    # ====================================================
    # Step 1: Generate the omega matrix (sum strictly equals total_edges)
    # ====================================================

    inter_strength = (1 - intra_strength) / (num_communities - 1)
    total_edges = avg_degree * n
    # Initialize the omega matrix
    omega = np.zeros((num_communities, num_communities))
    for i in range(num_communities):
        for j in range(num_communities):
            if i == j:
                omega[i][j] = intra_strength
            else:
                omega[i][j] = inter_strength

    # Normalize and convert to integer
    omega = (omega / omega.sum() * total_edges).astype(int)
    # Adjust the sum error
    diff = total_edges - omega.sum()
    omega[0][0] += diff  # Compensate the error to the first element

    # ====================================================
    # Step 2: Generate k1 (sum of node degrees = total_edges)
    # ====================================================
    k1 = {}
    remaining = total_edges
    for i in range(n):
        if i == n - 1:
            k1[i] = remaining  # Assign remaining degrees to the last node
        else:
            max_possible = min(remaining - (n - i - 1), 2 * avg_degree)  # Ensure at least 1 degree for subsequent nodes
            k1[i] = random.randint(1, max_possible)
            remaining -= k1[i]

    # ====================================================
    # Step 3: Generate k2 (sum of edge sizes = total_edges, and each edge size ≥2)
    # ====================================================
    m = total_edges // 3
    edge_sizes = []
    remaining_edges = total_edges

    # Generate all edges until remaining edges ≤ 1
    while remaining_edges > 1:
        # Dynamically calculate the maximum possible size
        max_size = min(5, remaining_edges)
        min_size = 2 if len(edge_sizes) < m-1 else max(2, remaining_edges)

        size = random.randint(min_size, max_size)
        edge_sizes.append(size)
        remaining_edges -= size

    # Handle the remaining quantity (ensure the sum is strictly equal to total_edges)
    if remaining_edges > 0:
        edge_sizes[-1] += remaining_edges  # Merge into the previous edge

    # Assign to k2
    k2 = {i: edge_sizes[i] for i in range(len(edge_sizes))}

    # ====================================================
    # Step 4: Verify sum consistency
    # ====================================================
    print("Sum of node degrees:", sum(k1.values()))
    print("Sum of edge sizes:", sum(k2.values()))
    print("Sum of Omega matrix:", omega.sum())
    assert sum(k1.values()) == omega.sum() == sum(k2.values()), "Parameter sums are inconsistent!"

    # ====================================================
    # Step 5: Generate community assignments
    # ====================================================
    g1 = {i: random.randint(0, num_communities-1) for i in range(n)}
    g2 = {}
    for edge_id in k2:
        nodes_in_edge = random.sample(list(k1.keys()), k2[edge_id])
        community_counts = np.zeros(num_communities)
        for node in nodes_in_edge:
            community_counts[g1[node]] += 1
        g2[edge_id] = np.argmax(community_counts)

    # ====================================================
    # Step 6: Generate hypergraph
    # ====================================================
    H = xgi.dcsbm_hypergraph(k1, k2, g1, g2, omega)

    H.cleanup(connected=False)
    Y = list(g1.values())
    _weight = np.ones(len(H.edges))
    edges = list(H.edges.members())

    G = xgi.convert.to_graph(H)
    labels_dict = {node: label for node, label in enumerate(Y)}
    nx.set_node_attributes(G, labels_dict, 'label')
    file_path = r''
    try:
        # Save the graph as a GEXF file
        nx.write_gexf(G, file_path)
        print(f"The graph has been successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

    # Louvain
    partition = community_louvain.best_partition(G)
    louvain_modularity = community_louvain.modularity(partition, G)
    louvain_labels = [partition[i] for i in G.nodes]
    true_labels = [g1[i] for i in G.nodes]
    nmi_louvain = normalized_mutual_info_score(true_labels, louvain_labels)

    # Girvan-Newman
    communities_gn = next(girvan_newman(G))
    community_list_gn = [list(c) for c in communities_gn]
    gn_modularity = nx.algorithms.community.quality.modularity(G, community_list_gn)

    gn_labels = [0]*n
    for i, comm in enumerate(community_list_gn):
        for node in comm:
            gn_labels[node] = i
    gn_nmi = normalized_mutual_info_score(Y, gn_labels)

    # LPA
    community_list_lpa = list(label_propagation_communities(G))
    lpa_modularity = nx.algorithms.community.quality.modularity(G, community_list_lpa)

    lpa_labels = [0]*n
    for i, comm in enumerate(community_list_lpa):
        for node in comm:
            lpa_labels[node] = i
    lpa_nmi = normalized_mutual_info_score(Y, lpa_labels)

    # Infomap
    im = Infomap()
    for u, v in G.edges():
        im.add_link(u, v)
    im.run()
    communities_infomap = {}
    for node in im.tree:
        if node.is_leaf:
            communities_infomap.setdefault(node.module_id, []).append(node.node_id)

    community_list_infomap = list(communities_infomap.values())

    mod_infomap = nx.algorithms.community.quality.modularity(G, community_list_infomap)

    infomap_labels = [0]*n
    for comm_id, comm in enumerate(community_list_infomap):
        for node in comm:
            infomap_labels[node] = comm_id
    infomap_nmi = normalized_mutual_info_score(Y, infomap_labels)

    # Walktrap
    G_ig = ig.Graph.from_networkx(G)
    walktrap = G_ig.community_walktrap()
    clusters = walktrap.as_clustering()
    community_list_walktrap = [list(cluster) for cluster in clusters]
    mod_walktrap = nx.algorithms.community.quality.modularity(G, community_list_walktrap)

    walktrap_labels = [0]*n
    for comm_id, comm in enumerate(community_list_walktrap):
        for node in comm:
            walktrap_labels[node] = comm_id
    walktrap_nmi = normalized_mutual_info_score(Y, walktrap_labels)

if __name__ == '__main__':
    main()
