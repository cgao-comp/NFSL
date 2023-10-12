import numpy as np
import sklearn
import args as args
from os.path import join as pjoin
import os
import networkx as nx
from tqdm import tqdm



class DataReader_snapshot():
    def __init__(self,
                 rumor_cascade,  
                 rnd_state=None,
                 folds=10,
                 union_graph_inf = None):

        self.data = {}
        self.rnd_state = rnd_state
        self.rumor_extend_cascade = rumor_cascade
        self.union_graph_inf = union_graph_inf

        self.graph_nodes = [len(g.nodes) for g in rumor_cascade]
        self.max_nodes = max(self.graph_nodes)
        self.graph_num = len(self.graph_nodes)

        
        train_ids, test_ids = split_ids(rnd_state.permutation(self.graph_num), folds=folds)  

        
        splits = []
        for fold in range(
                len(train_ids)):  
            splits.append({'train': train_ids[fold],
                           'test': test_ids[fold]})
        self.data['splits'] = splits
        self.generate_data()

        print()
        print()

    def generate_data(self):
        adj_list = []
        targets = []
        snapshots = []

        
        features_T = {f"features_T{i}": [] for i in range(5)}

        for g in tqdm(self.rumor_extend_cascade):
            
            uid_to_index = {uid: idx for idx, uid in enumerate(g.nodes)}

            
            adj_matrix = np.zeros((len(g.nodes), len(g.nodes)))
            for u, v in g.edges():
                if u == v:
                    continue
                adj_matrix[uid_to_index[u], uid_to_index[v]] = 1
                adj_matrix[uid_to_index[v], uid_to_index[u]] = 1
            adj_list.append(adj_matrix)

            
            times = nx.get_node_attributes(g, 'time')
            source_uid = min(times, key=times.get)  
            source_idx = uid_to_index[source_uid]

            
            one_hot_target = np.zeros(len(g.nodes))
            one_hot_target[source_idx] = 1
            targets.append(one_hot_target)

            
            unique_times = np.unique(list(times.values()))
            random_times = np.random.choice(unique_times, size=min(5, len(unique_times)), replace=False)
            random_times = sorted(random_times)

            
            if len(random_times) < 5:
                missing_count = 5 - len(random_times)
                random_times = np.concatenate((random_times, [max(unique_times)] * missing_count))

            snapshot_for_g = []
            for t_idx, t in enumerate(sorted(random_times)):  
                infected_nodes_at_t = [uid_to_index[node] for node, time in times.items() if time <= t]
                snapshot = np.zeros(len(g.nodes))
                for node_idx in infected_nodes_at_t:
                    snapshot[node_idx] = 1
                snapshot_for_g.append(snapshot)

                
                def get_node_features_for_snapshot(node, g, union_graph_inf):
                    """
                    Computes the required 12 attributes for a given node in a snapshot.
                    """
                    
                    features = list(union_graph_inf.nodes[node]['norm_profile_list'])

                    
                    
                    
                    
                    
                    
                    
                    
                    

                    
                    
                    infected_neighbors = [n for n in g.neighbors(node) if g.nodes[n]['time'] != 999999]
                    uninfected_neighbors = [n for n in g.neighbors(node) if g.nodes[n]['time'] == 999999]
                    max_infected_neighbors = max([len(list(g.neighbors(n))) for n in g.nodes()])

                    
                    infected_ratio = len(infected_neighbors) / len(list(g.neighbors(node))) if list(
                        g.neighbors(node)) else 0

                    features.append(
                        len(infected_neighbors) / max_infected_neighbors)  
                    features.append(
                        len(uninfected_neighbors) / max_infected_neighbors)  
                    features.append(infected_ratio)  
                    features.append(1 - infected_ratio)

                    
                    is_infected = 1 if g.nodes[node]['time'] != 999999 else 0
                    features.append(is_infected)
                    features.append(1 - is_infected)

                    return features

                
                
                node_info = {}
                for node in g.nodes():
                    neighbors = list(g.neighbors(node))
                    infected_neighbors = [n for n in neighbors if g.nodes[n]['time'] != 999999]
                    uninfected_neighbors = [n for n in neighbors if g.nodes[n]['time'] == 999999]
                    node_info[node] = {
                        'neighbors': neighbors,
                        'infected_neighbors': infected_neighbors,
                        'uninfected_neighbors': uninfected_neighbors,
                        'infected_ratio': len(infected_neighbors) / len(neighbors) if neighbors else 0
                    }
                max_infected_neighbors = max([len(info['neighbors']) for info in node_info.values()])

                def get_node_features_for_snapshotv2(node, g, union_graph_inf, node_info, max_infected_neighbors):
                    
                    features = list(union_graph_inf.nodes[node]['norm_profile_list'])

                    
                    features.append(len(node_info[node]['infected_neighbors']) / max_infected_neighbors)
                    features.append(len(node_info[node]['uninfected_neighbors']) / max_infected_neighbors)
                    features.append(node_info[node]['infected_ratio'])
                    features.append(1 - node_info[node]['infected_ratio'])

                    
                    is_infected = 1 if g.nodes[node]['time'] != 999999 else 0
                    features.append(is_infected)
                    features.append(1 - is_infected)

                    return features

                node_features_for_snapshot = []
                for uid in uid_to_index:
                    
                    features = get_node_features_for_snapshotv2(uid, g, self.union_graph_inf, node_info,
                                                              max_infected_neighbors)
                    
                    node_features_for_snapshot.append(features)
                features_T[f"features_T{t_idx}"].append(node_features_for_snapshot)

            snapshots.append(snapshot_for_g)

        self.data['adj_list'] = np.array(adj_list)
        self.data['targets'] = np.array(targets)
        self.data['snapshots'] = np.array(snapshots)

        for t_idx in range(5):
            self.data[f"features_T{t_idx}"] = np.array(features_T[f"features_T{t_idx}"])


    def generate_data_nofeature(self):
        adj_list = []
        targets = []
        snapshots = []

        for g in self.rumor_extend_cascade:
            
            uid_to_index = {uid: idx for idx, uid in enumerate(g.nodes)}

            
            adj_matrix = np.zeros((len(g.nodes), len(g.nodes)))
            for u, v in g.edges():
                if u == v:
                    continue
                adj_matrix[uid_to_index[u], uid_to_index[v]] = 1
                adj_matrix[uid_to_index[v], uid_to_index[u]] = 1
            adj_list.append(adj_matrix)

            
            times = nx.get_node_attributes(g, 'time')
            source_uid = min(times, key=times.get)  
            source_idx = uid_to_index[source_uid]

            
            one_hot_target = np.zeros(len(g.nodes))
            one_hot_target[source_idx] = 1
            targets.append(one_hot_target)

            
            unique_times = np.unique(list(times.values()))
            random_times = np.random.choice(unique_times, size=min(5, len(unique_times)), replace=False)
            random_times = sorted(random_times)

            
            if len(random_times) < 5:
                missing_count = 5 - len(random_times)
                random_times = np.concatenate((random_times, [max(unique_times)] * missing_count))

            snapshot_for_g = []
            for t in sorted(random_times):  
                infected_nodes_at_t = [uid_to_index[node] for node, time in times.items() if time <= t]
                snapshot = np.zeros(len(g.nodes))
                for node_idx in infected_nodes_at_t:
                    snapshot[node_idx] = 1
                snapshot_for_g.append(snapshot)

            snapshots.append(snapshot_for_g)

        self.data['adj_list'] = np.array(adj_list)
        self.data['targets'] = np.array(targets)
        self.data['snapshots'] = np.array(snapshots)


def split_ids(ids, folds=10):
    
    
    
    
    
    
    
    
    
    
    
    n = len(ids)
    stride = int(np.ceil(n / float(folds)))  
    test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
    assert np.all(
        np.unique(np.concatenate(test_ids)) == sorted(ids)), 'some graphs are missing in the test sets'
    assert len(test_ids) == folds, 'invalid test sets'
    train_ids = []
    for fold in range(folds):
        train_ids.append(np.array(
            [e for e in ids if e not in test_ids[fold]]))  
        assert len(train_ids[fold]) + len(test_ids[fold]) == len(
            np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

    return train_ids, test_ids