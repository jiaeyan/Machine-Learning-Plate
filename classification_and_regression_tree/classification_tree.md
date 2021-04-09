### Model  
A model to predict a __DISCRETE__ class by using info of __DEICRETE__ valued features. 
* internal node: 
  * represents a feature type (splitting variable) and one of its values (splitting point)  
  * splits the data into two parts (two children nodes), one with _== splitting point_ and another with _!= splitting point_ _(while they may not belong to the same class)_  
* terminal node: 
  * represents a __DISCRETE__ predicted class based on the feature and value path  
* path:
  * each path from root to terminal in the tree represents a set of __if-then rules__, which is made of by a set of values from different features
  * each instance is covered by one of the paths

### Steps

* Feature selection
  Choose the one that has the the smallest __GINI INDEX__ in two children. This process is recursive. Once a feature is used, __as long as it has other unused feature values, it's still valid in next feature selection iteration__ (because that feature still has new info to dig), _this is different from ID3 and C4.5, whcih consumes all feature values at once and discarded in next feature selection iteration_:
    * __GINI INDEX__ (the uncertainty of the data set, $k$=class number): $Gini(D)=Gini(p)=\sum_{k=1}^{K}p_{k}(1-p_{k})=1-\sum_{k=1}^{K}p_{k}^2 = 1 - \sum_{k=1}^{K} \left( \frac{|C_k|} {|D|} \right)^2$, 
    * Assume we choose feature $A$ and its value $a$ as splitting spot for current node, then separate the data into: $D_1 = \{(x,y)\in D|A(x)=a \}$, $\quad D_{2}=D-D_{1}$
    * So the whole GINI INDEX of the whole dataset is: $Gini(D,A=a)=\frac{|D_{1}|}{|D|}Gini(D_{1})+\frac{|D_{2}|}{|D|}Gini(D_{2})$
    * Iterate over all features and all their values, choose the one that minimizes Gini index.
* Tree generation
  * Once a splitting variable and point chosen, splt data and assign them to its two children node respectively
  * Each child chooses the next best splitting variable and value, generates its children
  * Recursion...
* Tree prune  

 


