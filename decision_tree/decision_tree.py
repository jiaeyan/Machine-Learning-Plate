from math import log2
from collections import Counter
from collections import defaultdict
import numpy as np
from sklearn.metrics import classification_report

np.random.seed(42)

class DecisionTree:

    class Node:
        def __init__(self, is_terminal=False, label=None, feature_id=None):
            # properties of internal/feature nodes
            self.feature_id = feature_id
            self.children = {}

            # properties of terminal/label nodes
            self.is_terminal = is_terminal
            self.label = label


    def __init__(self, info_gain_threshold=0.1, mode='id3'):
        self.info_gain_threshold = info_gain_threshold
        self.mode = mode
        self.tree = None

    def data_entropy(self, Y):
        Y_count = Counter(Y)
        return -np.sum((count/len(Y)) * log2(count/len(Y)) for count in Y_count.values())

    def cond_entropy(self, X, Y, feature_id):
        F_Y_dict = defaultdict(list)
        for x, y in zip(X, Y):
            F_Y_dict[x[feature_id]].append(y)
        return np.sum((len(y_list)/len(Y)) * self.data_entropy(y_list) for y_list in F_Y_dict.values())
    
    def f_data_entropy(self, X, Y, feature_id):
        F_Y_dict = defaultdict(list)
        for x, y in zip(X, Y):
            F_Y_dict[x[feature_id]].append(y)
        return -np.sum((len(y_list)/len(Y)) * log2(len(y_list)/len(Y)) for y_list in F_Y_dict.values())

    def info_gain(self, data_entropy, cond_entropy):
        return data_entropy - cond_entropy
    
    def info_gain_ratio(self, data_entropy, cond_entropy, f_data_entropy):
        return (data_entropy - cond_entropy) / f_data_entropy
    
    def train(self, X, Y):
        self.tree = self.generate_tree(X, Y)

        # prune the tree

    def generate_tree(self, X, Y):

        # if all labels are the same, make a terminal node
        if len(set(Y)) == 1:
            return self.Node(is_terminal=True, label=Y[0])

        # if no selectable feature left, make a terminal node
        if len(X[0]) == 0:
            return self.Node(is_terminal=True, label=Counter(Y).most_common()[0][0])

        # Step 1: select the feature with the highest info gain
        info_gain, best_feature_id = self.select_feature(X, Y)

        # if info gain is too small, make a terminal node
        if info_gain <= self.info_gain_threshold:
            return self.Node(is_terminal=True, label=Counter(Y).most_common()[0][0])
        
        # Step 2: grow the tree
        node = self.expand_tree(X, Y, best_feature_id)

        return node


    def select_feature(self, X, Y):
        d_ent = self.data_entropy(Y)
        info_gain_list = []

        for feature_id in range(len(X[0])):
            cond_ent = self.cond_entropy(X, Y, feature_id)

            if self.mode == 'id3':
                info_gain_list.append((self.info_gain(d_ent, cond_ent), feature_id))
            elif self.mode == 'c4.5':
                f_d_ent = self.f_data_entropy(X, Y, feature_id)
                info_gain_list.append((self.info_gain_ratio(d_ent, cond_ent, f_d_ent), feature_id))

        return max(info_gain_list)
        

    def expand_tree(self, X, Y, feature_id):
        node = self.Node(feature_id=feature_id)
        best_f_value_set = set(X[:, feature_id])

        for f in best_f_value_set:
            sub_X, sub_Y = zip(*[(np.delete(x, feature_id), y) for x, y in zip(X, Y) if x[feature_id] == f])
            sub_tree = self.generate_tree(np.array(sub_X), np.array(sub_Y))
            node.children[f] = sub_tree
        
        return node

    def prune_tree(self, X, Y):
        pass

    def predict(self, x):
        node = self.tree
        while not node.is_terminal:
            node = node.children[x[node.feature_id]]
        return node.label

    def score(self, X, Y):
        Y_pred = [self.predict(x) for x in X]
        print(classification_report(Y, Y_pred))

def generate_data():
    from sklearn.model_selection import train_test_split
    data = np.array([['青年', '否', '否', '一般', '否'],
            ['青年', '否', '否', '好', '否'],
            ['青年', '是', '否', '好', '是'],
            ['青年', '是', '是', '一般', '是'],
            ['青年', '否', '否', '一般', '否'],
            ['中年', '否', '否', '一般', '否'],
            ['中年', '否', '否', '好', '否'],
            ['中年', '是', '是', '好', '是'],
            ['中年', '否', '是', '非常好', '是'],
            ['中年', '否', '是', '非常好', '是'],
            ['老年', '否', '是', '非常好', '是'],
            ['老年', '否', '是', '好', '是'],
            ['老年', '是', '否', '好', '是'],
            ['老年', '是', '否', '非常好', '是'],
            ['老年', '否', '否', '一般', '否'],
            ])
    X = data[:, :-1]
    Y = data[:, -1]
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y)
    print('Train data shape', X_train.shape)
    print('Validation data shape', X_val.shape)
    return X_train, X_val, Y_train, Y_val

if __name__ == '__main__':
    X_train, X_val, Y_train, Y_val = generate_data()
    dt = DecisionTree()
    dt.train(X_train, Y_train)
    dt.score(X_val, Y_val)
