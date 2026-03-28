#!/usr/bin/env python3
"""random_forest2 - Random Forest with bootstrapping."""
import sys, random, math, collections
def entropy(labels):
    n=len(labels); counts=collections.Counter(labels)
    return -sum((c/n)*math.log2(c/n) for c in counts.values() if c>0)
def best_split(data, labels, features):
    best_gain, best_f, best_val = -1, None, None
    base_ent = entropy(labels)
    for f in features:
        vals = sorted(set(row[f] for row in data))
        for v in vals:
            left_l = [labels[i] for i in range(len(data)) if data[i][f]<=v]
            right_l = [labels[i] for i in range(len(data)) if data[i][f]>v]
            if not left_l or not right_l: continue
            gain = base_ent - (len(left_l)*entropy(left_l)+len(right_l)*entropy(right_l))/len(labels)
            if gain > best_gain: best_gain=gain; best_f=f; best_val=v
    return best_f, best_val
class TreeNode:
    def __init__(self, label=None, feature=None, threshold=None, left=None, right=None):
        self.label=label; self.feature=feature; self.threshold=threshold; self.left=left; self.right=right
def build_tree(data, labels, features, depth=0, max_depth=5):
    if len(set(labels))==1: return TreeNode(label=labels[0])
    if depth>=max_depth or not features: return TreeNode(label=collections.Counter(labels).most_common(1)[0][0])
    subset = random.sample(features, max(1,int(len(features)**0.5)))
    f, v = best_split(data, labels, subset)
    if f is None: return TreeNode(label=collections.Counter(labels).most_common(1)[0][0])
    l_idx = [i for i in range(len(data)) if data[i][f]<=v]
    r_idx = [i for i in range(len(data)) if data[i][f]>v]
    if not l_idx or not r_idx: return TreeNode(label=collections.Counter(labels).most_common(1)[0][0])
    left = build_tree([data[i] for i in l_idx],[labels[i] for i in l_idx],features,depth+1,max_depth)
    right = build_tree([data[i] for i in r_idx],[labels[i] for i in r_idx],features,depth+1,max_depth)
    return TreeNode(feature=f, threshold=v, left=left, right=right)
def predict_tree(node, x):
    if node.label is not None: return node.label
    return predict_tree(node.left if x[node.feature]<=node.threshold else node.right, x)
class RandomForest:
    def __init__(self, n_trees=10, max_depth=5): self.n_trees=n_trees; self.max_depth=max_depth; self.trees=[]
    def fit(self, X, Y):
        n=len(X); features=list(range(len(X[0])))
        for _ in range(self.n_trees):
            idx = [random.randint(0,n-1) for _ in range(n)]
            self.trees.append(build_tree([X[i] for i in idx],[Y[i] for i in idx],features,max_depth=self.max_depth))
    def predict(self, x):
        votes = [predict_tree(t, x) for t in self.trees]
        return collections.Counter(votes).most_common(1)[0][0]
if __name__=="__main__":
    random.seed(42)
    X = [[random.gauss(-2,1),random.gauss(-2,1)] for _ in range(50)] + [[random.gauss(2,1),random.gauss(2,1)] for _ in range(50)]
    Y = [0]*50 + [1]*50
    rf = RandomForest(n_trees=20); rf.fit(X, Y)
    correct = sum(1 for x,y in zip(X,Y) if rf.predict(x)==y)
    print(f"Random Forest Accuracy: {correct}/100 = {correct}%")
