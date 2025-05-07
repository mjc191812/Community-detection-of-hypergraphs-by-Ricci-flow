import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cvx
import xgi
import sys
import itertools
import pandas as pd
import concurrent.futures
from sklearn import metrics
import networkx as nx
from sklearn import preprocessing, metrics
import random
import os
import community as community_louvain
from networkx.algorithms import community
from networkx.algorithms.community import girvan_newman, label_propagation_communities
from infomap import Infomap
import igraph as ig
from sklearn.metrics import normalized_mutual_info_score
alpha=0.5
s=0.1




def min_where(dicty, where):
    min_val = np.inf
    for key in dicty.keys():
        if where[key]:
            if dicty[key] < min_val:
                min_val = dicty[key]
            else:
                pass
        else:
            pass
    return min_val

def single_source_shortest_path_length(H, source):

    # 1. Mark all nodes unvisited.
    is_unseen = dict()
    for node in H.nodes:
        is_unseen[node] = True
    n_unseen = len(H.nodes)

    # 2. Assign to every node a tentative distance value.
    dists = dict()
    for node in H.nodes:
        dists[node] = np.inf
    dists[source] = 0
    is_unseen[source] = False
    n_unseen -= 1
    current = source

    # 3. Consider all of unvisited neighbors of current node and calculate their tentative distances to the current node.
    stop_condition = False
    while not stop_condition:
        for ngb in H.nodes.neighbors(current):
            if is_unseen[ngb]:
                for edge in list(H.edges.members()):
                    if current in edge and ngb in edge:
                        weight=_weight[list(H.edges.members()).index(edge)]
                
                #print(current,ngb,weight)
                #for key in _dict.keys():
                    # if current in key and ngb in key:
                    #     weight=_dict[key]
                    #     break
                increment =  weight # increment = weight(ngb, current) if weighted
                new_dist = dists[current] + increment
                if new_dist < dists[ngb]:
                    dists[ngb] = new_dist
                else:
                    pass
            else:
                pass

        # 4. Mark the current node as visited and remove it from the unvisited set.
        is_unseen[current] = False
        n_unseen -= 1

        # 5. Check for stop condition.
        stop_condition = (n_unseen == 0) or (
            min_where(dists, is_unseen) == np.inf
        )

        # 6. Otherwise, select the unvisited node that is marked with the smallest tentative distance, set it as the new current node, and go back to step 3.
        min_val = np.inf
        argmin = current
        for node in dists.keys():
            if is_unseen[node]:
                if dists[node] < min_val:
                    min_val = dists[node]
                    argmin = node
                else:
                    pass
            else:
                pass
        current = argmin

    return dists


def shortest_path_length(H):
 

    for source in H.nodes:
        dists = single_source_shortest_path_length(H, source)
        yield (source, dists)



def _get_all_pairs_shortest_path(H):
    # Construct the all pair shortest path lookup
    lengths = dict(shortest_path_length(H))
    return lengths








# calculate the distribution for a given node
def _get_single_node_neighbors_distributions(node):
    
    
    dist = np.zeros(len(H.nodes))
    
    x = H.nodes.degree.aslist()
    sum_w=0
    A=list(H.edges.members())
    for edge in A:
        if node in edge:
            sum_w+=_weight[A.index(edge)]
            
    if sum_w==0:
        return [1], [node]
    for edge in A:
        if node in edge:
            B=list(edge)
            now=(1-alpha)* _weight[A.index(edge)] / sum_w
            for vertex in edge:
                if vertex!= node:
                    
                    y = H.edges.order.aslist()                
                    dist[vertex] += (now * 1 / y[A.index(edge)])
                    #print("node",node,"vertex",vertex,"degree",x[B.index(node)],"order",y[A.index(edge)])
    dist[node] += alpha
    
    distribution_keys = np.arange(len(dist))
    distribution_values = dist
    return distribution_values, distribution_keys



def _get_edge_density_distributions():
    densities = dict()
    for x in H.nodes():
        densities[x] = _get_single_node_neighbors_distributions(x)

def _optimal_transportation_distance( x, y, d):
    rho = cvx.Variable((len(y), len(x)))  

    # objective function d(x,y) * rho * x, need to do element-wise multiply here
    obj = cvx.Minimize(cvx.sum(cvx.multiply(np.multiply(d.T, x.T), rho)))

    # \sigma_i rho_{ij}=[1,1,...,1]
    source_sum = cvx.sum(rho, axis=0, keepdims=True)
    constrains = [rho @ x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
    prob = cvx.Problem(obj, constrains)

    m = prob.solve(solver="ECOS")  # change solver here if you want
    return m

def _distribute_densities(source, target):
    # Append source and target node into weight distribution matrix x, y
    x, source_topknbr = _get_single_node_neighbors_distributions(source)
    y, target_topknbr = _get_single_node_neighbors_distributions(target)
    d = []
    for src in source_topknbr:
        tmp = []
        for tgt in target_topknbr:
            if lengths[src][tgt] != float('inf'):
                tmp.append(lengths[src][tgt])
            else:
                tmp.append(9999)
        d.append(tmp)
    d = np.array(d)
    x = np.array(x)
    y = np.array(y)
    return x, y, d

def _compute_ricci_curvature_single_edge(edge):
    W = 0
    edge_list = list(edge)
    if len(edge_list) < 2:
        return W
    for i in range(len(edge_list)):
        for j in range(i + 1, len(edge_list)):
            x, y, d = _distribute_densities(edge_list[i], edge_list[j])
            m = _optimal_transportation_distance(x, y, d)
            # print("between node", edge_list[i], "and node", edge_list[j], "the optimal transportation distance is", m)
            W += m
    return 2 * W / (len(edge_list) * (len(edge_list) - 1))
    # Now returns Wasserstein distance instead of Ricci curvature, which is sufficient for Ricci flow weight updating process

def _compute_ricci_curvature_all_edges():
    ricci_curvatures = []
    for edge in H.edges.members():
        ricci_curvatures.append(_compute_ricci_curvature_single_edge(edge))
    return ricci_curvatures

def _compute_ricci_curvature_all_edges_parallel():
    ricci_curvatures = [None] * len(edges)  # Create a list with the same length as the original edges to store results
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks to the thread pool, along with the index of the edge
        future_to_edge_index = {executor.submit(_compute_ricci_curvature_single_edge, edge): index
                                for index, edge in enumerate(H.edges.members()) if len(edge) > 1}

        for future in concurrent.futures.as_completed(future_to_edge_index):
            index = future_to_edge_index[future]
            try:
                ricci_curvature = future.result()
                ricci_curvatures[index] = ricci_curvature  # Store the result in the correct position based on the index
            except Exception as exc:
                print(f'Edge {edges[index]} generated an exception: {exc}')

    return ricci_curvatures



def update_weight_single_edge(edge):
    _sum=0
    edge_list = list(edge)
    if len(edge_list)<2:
        return 0
    for i in range(len(edge_list)):
            for j in range(i + 1, len(edge_list)):
                _sum+=lengths[edge_list[i]][edge_list[j]]
    return 2*_sum/(len(edge_list)*(len(edge_list)-1))

def update_weight_all_edges():
    ricci_curvatures = _compute_ricci_curvature_all_edges()
    edge_count = len(list(H.edges.members()))
    if len(_weight) != edge_count or len(ricci_curvatures) != edge_count:
        raise ValueError("The lengths of _weight and ricci_curvatures must match the number of edges in the hypergraph.")
    edge_index_map = {tuple(sorted(edge)): index for index, edge in enumerate(H.edges.members())}
    for edge in H.edges.members():
        edge_tuple = tuple(sorted(edge))
        index = edge_index_map[edge_tuple]
        _weight[index] += s * (ricci_curvatures[index] - update_weight_single_edge(edge))


def update_weight_all_edges_parallel():
    ricci_curvatures = _compute_ricci_curvature_all_edges_parallel()
    edge_index_map = {tuple(sorted(edge)): index for index, edge in enumerate(H.edges.members())}
    for edge in H.edges.members():
        edge_tuple = tuple(sorted(edge))
        index = edge_index_map[edge_tuple]
        if len(edge) > 1:
            _weight[index] +=s*( ricci_curvatures[index] - update_weight_single_edge(edge))





def  check_accuracy(G, H, weight="weight", Ground_truth_label="value"):
   

    N_nodes = len(H.nodes)
    N_edges = len(H.edges)

    hyp_curvs = ricci_curvatures

    hyp_curvs = [curv for curv in hyp_curvs if curv is not None and not (isinstance(curv, float) and (curv == float('inf') or curv == float('-inf')))]    
    if hyp_curvs:
        maxw = max(hyp_curvs)
        minw = min(hyp_curvs)
    else:
        maxw = 1000
        minw = 0
    
    
    maxw=min(maxw,minw+100)
    cutoff_range = np.arange(maxw, minw , -0.01)
 
    modularity, ari ,nmi = [], [], []
    for cut in cutoff_range:
        edge_keep_list = list(np.argwhere(hyp_curvs<cut).T[0])
        new_edges =[]
        for k in edge_keep_list:
            new_edges.append(edges[k])
        for node in range(N_nodes):
            new_edges.append([node])
        new_HG = xgi.Hypergraph(new_edges)
        G=xgi.convert.graph.to_graph(new_HG)
        if G.number_of_edges() == 0:
            print("No edges left in the graph. Exiting the loop.")
            break
        c_communities=list(nx.connected_components(G))

        clustering_labels=-1+np.zeros(len(Ground_truth_label))
        for k, component in enumerate(c_communities):
            clustering_labels[list(component)]=k

        NMI = metrics.normalized_mutual_info_score(Ground_truth_label,clustering_labels)
        ARI=metrics.adjusted_rand_score(Ground_truth_label,clustering_labels)
        Modularity=(nx.community.modularity(G, c_communities))
        modularity.append(Modularity)
        ari.append(ARI)
        nmi.append(NMI)
        
    #print(f"max_modularity={max(modularity)}, max_ARI={max(ari)}, max_NMI={max(nmi)}")
    return max(modularity), max(ari), max(nmi)







 
def draw_graph(G, clustering_label="club"):
    """
    A helper function to draw a nx graph with community.
    """
    node_color = clustering_label   
    pos=nx.spring_layout(G)
    nx.draw_spring(G,nodelist=G.nodes(),
                   node_color=node_color,
                   cmap=plt.cm.rainbow,
                   alpha=0.8)
    plt.show()







def main():
    global lengths, H, _weight, Y, edges,ricci_curvatures


    H = xgi.Hypergraph()
    HG= pd.read_pickle(r'')
    Y=pd.read_pickle(r'')
    Y=np.array(Y)
    edges = list(HG.values())
    n = len(Y)
    _weight = np.ones(len(edges))
    H.add_edges_from(edges)



    G=xgi.convert.graph.to_graph(H)
    
    # Louvain
    partition = community_louvain.best_partition(G)
    louvain_modularity = community_louvain.modularity(partition, G)
    louvain_labels = [partition[i] for i in G.nodes]  
    true_labels = [Y[i] for i in G.nodes]  
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
    

        
    # 创建 Infomap 对象
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
    

    G_ig = ig.Graph.from_networkx(G)

    #  Walktrap 
    walktrap = G_ig.community_walktrap()
    clusters = walktrap.as_clustering()  
    community_list_walktrap = [list(cluster) for cluster in clusters]
    mod_walktrap = nx.algorithms.community.quality.modularity(G, community_list_walktrap)
    
    walktrap_labels = [0]*n
    for comm_id, comm in enumerate(community_list_walktrap):
        for node in comm:
            walktrap_labels[node] = comm_id
    walktrap_nmi = normalized_mutual_info_score(Y, walktrap_labels)
    


    lengths = _get_all_pairs_shortest_path(H)
    iter=20
    for i in range(iter):
        ricci_curvatures = _compute_ricci_curvature_all_edges_parallel()    
        update_weight_all_edges()
        lengths = _get_all_pairs_shortest_path(H)
        G=xgi.convert.graph.to_graph(H)
    
    a,b,c=check_accuracy(G, H, weight="weight", Ground_truth_label=Y)
    # save_path = fr""
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # with open(save_path, "w", encoding="utf-8") as f:
    #     sys.stdout = f
        # print(f"Louvain Modularity: {louvain_modularity}")
        # print(f"Louvain NMI: {nmi_louvain}")
        # print(f"Girvan-Newman Modularity: {gn_modularity}")
        # print(f"Girvan-Newman NMI: {gn_nmi}")
        # print(f"LPA Modularity: {lpa_modularity}")
        # print(f"LPA NMI: {lpa_nmi}")
        # print(f"Infomap Modularity:", mod_infomap)
        # print(f"Infomap NMI: {infomap_nmi}")
        # print(f"Walktrap Modularity:", mod_walktrap)
        # print(f"Walktrap NMI: {walktrap_nmi}")
        # print(f"our_algorithm_Modularity:",a)
        # print(f"our_algorithm_NMI:",c)
        # sys.stdout = sys.__stdout__





if __name__ == '__main__':
    main()


