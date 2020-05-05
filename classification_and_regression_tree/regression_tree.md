### Model
A model to predict a __CONTINUOUS__ real value by using info of __CONTINUOUS__ features.  
It's a __BINARY__ tree, with:
* internal node: 
  * represents a feature type (splitting variable) and one of its values (splitting point)  
  * splits the data into two parts (two children nodes), one with _<= splitting point_ and another with _> splitting point_ _(while they may not belong to the same class)_   
  *  has a __TEMP__ predicted label, used to train the tree
* terminal node: 
  * represents a __CONTINUOUS__ predicted value based on the feature and value path  
  * all termonial nodes constitue the separation units in the feature space  
* path:
  * each path from root to terminal in the tree represents a set of __if-then rules__, which is made of by a set of values from different features
  * each instance is covered by one of the paths

### Steps

* Feature selection  
  Choose the one that minimizes the __MEAN SQUARE ERROR__ of data in two children. This process is recursive. Once a feature is used, __as long as it has other unused feature values, it's still valid in next feature selection iteration__ (because that feature still has new info to dig), _this is different from ID3 and C4.5, whcih consumes all feature values at once and discarded in next feature selection iteration_:
    * __Internal Node__: each internal node contains a set of data, the splitting point should make the sum of MSE of all nodes minimum: $\min \frac{1}{n} \sum_{i = 1}^{n} (f(\bm x_i) - y_i)^2$. Use derivative here, we know to reach this only when $f(x_{i}) = c_{m} = ave(y_i | \bm x_i \in leaf_m)$, where $c_{m}$ is the label of the node  
    * __Whole Tree Error__ (assume there are $m$ terminal nodes): $\min \frac{1}{n} \sum_{m = 1}^{M}\sum_{\bm x_i \in R_m} (c_m - y_i)^2$
    * __How to select__: iterate over all features, and iterate all valid feature values, find the one that splits the data like below: $\begin{aligned} R_1\{f, f\_val\} = {\bm x| \bm x^{(f)} \le f\_val} \\ R_2\{f, f\_val\} = {\bm x| \bm x^{(f)} \gt f\_val}\end{aligned}$ and minimizes this: $\min_{f, f\_val} \left[\min_{c_1} \sum_{\bm x_i \in R_1\{f, f\_val\}} (y_i - c_1)^2 + \min_{c_2} \sum_{\bm x_i \in R_2\{f, f\_val\}} (y_i - c_2)^2 \right]$, where $c_1 = ave(y_i | \bm x_i \in R_1\{f, f\_val\}$ and $c_2 = ave(y_i | \bm x_i \in R_2\{f, f\_val\}$.
    * __How to label__: mean of all data labels in the node
* Tree generation
  * Once a splitting variable and point chosen, splt data and assign them to its two children node respectively
  * Each child chooses the next best splitting variable and value, generates its children
  * Recursion...
* Tree prune
 


