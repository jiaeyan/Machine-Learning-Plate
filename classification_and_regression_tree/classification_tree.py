import sys
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report

np.random.seed(42)

class Node:

    def __init__(self, is_terminal=False, label=None, feature_id=None, feature_val=None):
        # properties of internal/feature nodes
        self.feature_id = feature_id
        self.feature_val = feature_val
        self.left = None
        self.right = None

        # properties of terminal/label nodes
        self.is_terminal = is_terminal
        self.label = label

class ClassificationTree:

    def __init__(self):
        self.tree = None
        self.features = None

    def train(self, X, Y):
        self.features = self.gather_features(X)
        self.tree = self.generate_tree(X, Y)
        # prune tree
    
    def gather_features(self, X):
        """
        Returns a feature dict which indicates what feature and which of its value is still valid in next feature selection iteration. 
        """
        return {feature_id: {feature_val: True for feature_val in set(X[:, feature_id])} for feature_id in range(len(X[0]))}
    
    def generate_tree(self, X, Y):
        # if all labels are the same, return a terminal node
        if len(set(Y)) == 1:
            return Node(is_terminal=True, label=Y[0])

        # Step 1: feature and split point selection
        feature_id, feature_val = self.select_feature(X, Y)

        # no good feature selected (e.g., chosen feature has only 1 value)
        if feature_id is None:
            return Node(is_terminal=True, label=Counter(Y).most_common()[0][0])

        # Step 2: split the binary tree
        node = self.expand_tree(X, Y, feature_id, feature_val)

        return node
    
    def select_feature(self, X, Y):
        gini_data = sys.maxsize
        best_feature_id = None
        best_feature_val = None

        for feature_id in range(len(X[0])):
            f_val_set = set(X[:, feature_id])
            for f_val in f_val_set:
                if self.features[feature_id][f_val]:
                    Y_left = Y[X[:, feature_id] == f_val]
                    gini_left = self.gini(Y_left)

                    Y_right = Y[X[:, feature_id] != f_val]
                    gini_right = self.gini(Y_right)

                    new_gini_data = (len(Y_left) / len(Y)) * gini_left + (len(Y_right) / len(Y)) * gini_right

                    if new_gini_data < gini_data:
                        gini_data = new_gini_data
                        best_feature_id = feature_id
                        best_feature_val = f_val

        if best_feature_val is not None:
            self.features[best_feature_id][best_feature_val] = False

        return best_feature_id, best_feature_val
    
    def gini(self, Y):
        Y_dict = Counter(Y)
        return 1.0 - sum(np.power(np.array(list(Y_dict.values())) / len(Y), 2))
    
    def expand_tree(self, X, Y, feature_id, feature_val):
        node = Node(feature_id=feature_id, feature_val=feature_val)

        X_left = X[X[:, feature_id] == feature_val]
        Y_left = Y[X[:, feature_id] == feature_val]
        node.left = self.generate_tree(X_left, Y_left)

        X_right = X[X[:, feature_id] != feature_val]
        Y_right = Y[X[:, feature_id] != feature_val]
        node.right = self.generate_tree(X_right, Y_right)

        return node
    
    def predict(self, x):
        node = self.tree

        while not node.is_terminal:
            if x[node.feature_id] <= node.feature_val:
                node = node.left
            else:
                node = node.right
        
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

if __name__ == "__main__":
    X_train, X_val, Y_train, Y_val = generate_data()
    ct = ClassificationTree()
    ct.train(X_train, Y_train)
    ct.score(X_val, Y_val)
