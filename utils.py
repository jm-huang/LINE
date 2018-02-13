import numpy as np
import networkx as nx
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def evaluate(embs, labels):
    x_train, x_test, y_train, y_test = \
        train_test_split(embs, labels, test_size=0.5, random_state=12345)
    scalar = preprocessing.StandardScaler()
    x_train = scalar.fit_transform(x_train)
    x_test = scalar.transform(x_test)
    clsf = LogisticRegression(penalty="l2")
    clsf.fit(x_train, y_train)

    train_pred = clsf.predict(x_train)
    train_acc = metrics.accuracy_score(y_train, train_pred)
    # train_report = metrics.classification_report(y_train, train_pred)

    test_pred = clsf.predict(x_test)
    test_acc = metrics.accuracy_score(y_test, test_pred)
    # test_report = metrics.classification_report(y_test, test_pred)

    return train_acc, test_acc


def get_labels(label_name):
    labels = np.loadtxt(label_name, dtype=int)
    labels = labels[np.argsort(labels[:, 0])][:, 1]
    return labels


class DataLoader:
    '''

    '''
    def __init__(self, graph_file, node_count):
        self.load_graph(graph_file, node_count)
        self.num_of_nodes = self.g.number_of_nodes()
        self.num_of_edges = self.g.number_of_edges()
        self.edges_raw = self.g.edges(data=True)
        self.nodes_raw = self.g.nodes(data=True)
        dis = np.array([attr['weight'] for _, _, attr in self.edges_raw], dtype=np.float32)
        self.edge_distribution = dis
        self.edge_distribution /= np.sum(self.edge_distribution)
        self.edge_sampling = AliasSampling(prob=self.edge_distribution)
        degree = np.array([self.g.degree(node, weight='weight') for node, _ in self.nodes_raw],
                          dtype=np.float32)
        self.node_negative_distribution = np.power(degree, 0.75)
        self.node_negative_distribution /= np.sum(self.node_negative_distribution)
        self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

        self.node_index = {}
        self.node_index_reversed = {}
        for index, (node, _) in enumerate(self.nodes_raw):
            self.node_index[node] = index
            self.node_index_reversed[index] = node
        self.edges = [(self.node_index[u], self.node_index[v]) for u, v, _ in self.edges_raw]

    def load_graph(self, graph_file, node_count):

        network = nx.DiGraph()
        network.add_nodes_from(range(node_count))

        with open(graph_file) as fid:
            for line in fid:
                vi, vj, w = line.strip().split()
                network.add_edge(int(vi), int(vj), weight=int(w))
            self.g = network

    def fetch_batch(self, batch_size=16, K=10, edge_sampling='atlas', node_sampling='atlas'):
        if edge_sampling == 'numpy':
            edge_batch_index = np.random.choice(self.num_of_edges,
                                                size=batch_size, p=self.edge_distribution)
        elif edge_sampling == 'atlas':
            edge_batch_index = self.edge_sampling.sampling(batch_size)
        elif edge_sampling == 'uniform':
            edge_batch_index = np.random.randint(0, self.num_of_edges, size=batch_size)
        u_i = []
        u_j = []
        label = []
        for edge_index in edge_batch_index:
            edge = self.edges[edge_index]
            if self.g.__class__ == nx.Graph:
                # important: second-order proximity is for directed edge
                if np.random.rand() > 0.5:
                    edge = (edge[1], edge[0])
            u_i.append(edge[0])
            u_j.append(edge[1])
            label.append(1)
            for i in range(K):
                while True:
                    if node_sampling == 'numpy':
                        negative_node = np.random.choice(self.num_of_nodes,
                                                         p=self.node_negative_distribution)
                    elif node_sampling == 'atlas':
                        negative_node = self.node_sampling.sampling()
                    elif node_sampling == 'uniform':
                        negative_node = np.random.randint(0, self.num_of_nodes)
                    if not self.g.has_edge(self.node_index_reversed[negative_node],
                                           self.node_index_reversed[edge[1]]):
                        break
                u_i.append(edge[0])
                u_j.append(negative_node)
                label.append(-1)
        return u_i, u_j, label

    def fetch_batch2(self, batch_size=16, K=10, edge_sampling='atlas', node_sampling='atlas'):
        '''
        speed up the fetch_batch() function by removing checking
        whether a negative edge is legal or not(sometimes a negative edge is postive edge
        Conclusion:
            1) about 20x than fetch_batch()
            2) uniform faster(1x) than atlas faster(10x) than numpy
        '''
        if edge_sampling == 'numpy':
            edge_batch_index = np.random.choice(self.num_of_edges,
                                                size=batch_size, p=self.edge_distribution)
        elif edge_sampling == 'atlas':
            edge_batch_index = self.edge_sampling.sampling(batch_size)
        elif edge_sampling == 'uniform':
            edge_batch_index = np.random.randint(0, self.num_of_edges, size=batch_size)

        size = batch_size * (K + 1)
        u_i = np.ones(size, int) * -1
        u_j = np.ones(size, int) * -1
        label = np.ones(size, int) * -10
        if node_sampling == 'numpy':
            negative_nodes = np.random.choice(self.num_of_nodes, size=size,
                                              p=self.node_negative_distribution)
        elif node_sampling == 'atlas':
            negative_nodes = self.node_sampling.sampling(size)
        elif node_sampling == 'uniform':
            negative_nodes = np.random.randint(0, self.num_of_nodes, size=size)

        for idx, edge_index in enumerate(edge_batch_index):
            edge = self.edges[edge_index]
            if self.g.__class__ == nx.Graph:
                # important: second-order proximity is for directed edge
                if np.random.rand() > 0.5:
                    edge = (edge[1], edge[0])
            start_idx = idx * (K + 1)
            end_idx = start_idx + K + 1
            u_i[start_idx: end_idx] = edge[0]
            u_j[start_idx] = edge[1]
            u_j[start_idx + 1: end_idx] = negative_nodes[start_idx + 1: end_idx]
            label[start_idx] = 1
            label[start_idx + 1: end_idx] = -1
        return u_i, u_j, label

    def embedding_mapping(self, embedding):
        return {node: embedding[self.node_index[node]] for node, _ in self.nodes_raw}


class AliasSampling:
    # Reference: https://en.wikipedia.org/wiki/Alias_method
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res
