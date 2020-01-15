###Model  
A model to describe/classify the training instances by features.  
It's a tree, with:
* internal node: 
  * it represents a feature value
  * it represents all instances so far that have this feature value _(while they may not belong to the same class)_  
* terminal node: 
  * it represents a class
  * it represents all instances so far that the model think SHOULD belong to this class _(while they may not belong to the same class)_. 
  * each terminal node constitutes a class probability distribution of this node, and its class is decided by the majority principle
  * all termonial nodes constitue the classification units in the feature space  
* path:
  * each path from root to terminal in the tree represents a set of __if-then rules__, which is made of by a set of values from different features
  * each path represents the definition of a class
  * all paths are mutually exclusive
  * each instance is covered by one of the paths

###Steps

* Feature selection
  How to decide which feature to use to classify? Choose the one that has the highest information gain / information gain ratio. This process is recursive, once a feature is used, it should be removed from next iteration:
    * __Data entroy__ (the uncertainty of the data set, $k$=class number): $H(D)=-\sum\limits_{k=1}^{K}\frac{|C_k|}{|D|}log_2 \frac{C_k}{D}$
    * __Condation entropy__ (the uncertainty of the data set after knowing the feature, $i$ = feature value number, $k$ = class number under that feature value $i$): $\begin{aligned}
    H(D|A) &= \sum\limits_{i=1}^{n}\frac{|D_i|}{|D|}H(D_i) \\ 
    &=-\sum\limits_{i=1}^{n}\frac{|D_i|}{|D|}\sum\limits_{n=1}^{K}\frac{|D_{ik}|}{|D_i|}log_2 \frac{D_{ik}}{D}
    \end{aligned}$
    * __Information gain__ (the amount of reduced uncertainty of the data set after knowing the feature): $g(D,A)=H(D)-H(D|A)$
    * __Data entropy under a feature__ (the uncertainty of the data set which is made of by one feature $i$): $H_A(D)=-\sum\limits_{i=1}^{n}\frac{|D_i|}{|D|}log_2 \frac{D_i}{D}$
    * __Information gain ratio__ (avoids the bias of information gain to the feature that has more values): $g_R(D,A)=\frac{g(D,A)}{H_A(D)}$
* Tree generation
  * ID3 -> information gain
  * C4.5 -> information gain ratio
* Tree prune
  * to avoid overfitting
  * Loss function: $C_\alpha(T)=\sum\limits_{t=1}^{|T|}N_tH_t(T)+\alpha|T|$
  * where $H_t(T)=-\sum\limits_{k}\frac{N_{tk}}{N_t}log\frac{N_{tk}}{N_t}$
  * $|T|$ = number of terminal nodes, $N$ = number of instances of that terminal node, $k$ = class number of the $N$ instances, $\alpha$ = parameter of model complexity, the bigger the simpler the tree is, the smaller the more complex the tree is
  * try all set of terminal nodes that have the same parent, cut them (combine them to their parent), compute loss function A, and if:
  * $C_\alpha(T_A)  \leq C_\alpha(T_B)$
  * then cut them, otherwise keep them. This process is recurisve to all the nodes in tree.
 


