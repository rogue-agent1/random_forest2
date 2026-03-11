#!/usr/bin/env python3
"""Random forest classifier — zero deps."""
import random,math
from collections import Counter

class DecisionStump:
    def __init__(self):self.feature=0;self.threshold=0;self.left=0;self.right=0
    def predict(self,x):return self.left if x[self.feature]<=self.threshold else self.right

class RandomForest:
    def __init__(self,n_trees=10,max_features=None):self.n_trees=n_trees;self.mf=max_features;self.trees=[]
    def fit(self,X,y):
        n=len(X);d=len(X[0]);self.mf=self.mf or max(1,int(d**0.5))
        for _ in range(self.n_trees):
            idx=[random.randint(0,n-1) for _ in range(n)]
            Xb=[X[i] for i in idx];yb=[y[i] for i in idx]
            feats=random.sample(range(d),min(self.mf,d))
            stump=self._fit_stump(Xb,yb,feats)
            self.trees.append(stump)
    def _fit_stump(self,X,y,feats):
        best=DecisionStump();best_score=-1
        for f in feats:
            vals=sorted(set(x[f] for x in X))
            for t in vals:
                left=[y[i] for i in range(len(X)) if X[i][f]<=t]
                right=[y[i] for i in range(len(X)) if X[i][f]>t]
                if not left or not right:continue
                score=max(Counter(left).values())/len(left)+max(Counter(right).values())/len(right)
                if score>best_score:best_score=score;best.feature=f;best.threshold=t;best.left=Counter(left).most_common(1)[0][0];best.right=Counter(right).most_common(1)[0][0]
        return best
    def predict(self,X):
        results=[]
        for x in X:
            votes=[t.predict(x) for t in self.trees]
            results.append(Counter(votes).most_common(1)[0][0])
        return results

def test():
    random.seed(42)
    X=[[random.gauss(1,1),random.gauss(1,1)] for _ in range(50)]+[[random.gauss(-1,1),random.gauss(-1,1)] for _ in range(50)]
    y=[1]*50+[0]*50
    rf=RandomForest(20);rf.fit(X,y)
    acc=sum(1 for p,t in zip(rf.predict(X),y) if p==t)/100
    assert acc>0.7
    print(f"Accuracy: {acc:.0%}");print("All tests passed!")
if __name__=="__main__":test()
