#### Main idea
In the feature space, the label of the K points that are the cloest to the input point have is the label to the input point.  

No explicit learning process, thus no likelihood, no cost function and no update rule.  

#### K number
If K is big, the model is simple, but very far noisy points could affect decision;
If K is small, the model is complex, easy to be overfitting;

#### Distance
Euclidean distance:
$L_2(x^{(i)}, x^{(j)}) = \sqrt{(\sum_{k=0}^d(x_k^{(i)} - x_k^{(j)})^2)}$

#### Vote
The more same labels in K points, the more reliable.