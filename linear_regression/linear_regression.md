#### Model  
Linear regression computes a real number value given an instance $x$ and a set of parameters $\theta$.  

$$h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3 + ... = \sum_{i=0}^d\theta_ix_i = \theta^Tx$$   

$\theta_0$ is generally referred as the bias/b.  

#### Likelihood  
Since a real number is predicted, rather than a probability, there is no likelihood for the data set.  

#### Loss function  
Least mean square cost function. This is what we want to minimize. We want to find all $\theta$ that minimize this; in other words, the optimal/minimum point of the convex (boat shape).  

In regression, local optimal = global optiaml.  


$$J(\theta) = \frac 12\sum_{i=0}^n(h_\theta(x^{(i)})-y^{(i)})^2$$  

#### Update rule  
If the derivative/切线斜率 > 0 on the convex, means $\theta$ could be smaller (move left to the convex to get closer to the optimal), thus minus; if < 0, means $\theta$ should be bigger (move right to the convex to get closer to the optimal), thus add; always in opposite conditions, thus minus the derivative.  

Derivative = the predicted error difference $(h_\theta(x^{(i)})-y^{(i)})$ * the corresponding feature $x_j^{(i)}$.  

single example single weight:  
$$\theta_j := \theta_j - \alpha(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$$  
single example all weights (__SGD__):  
$$\theta := \theta - \alpha(h_\theta(x^{(i)})-y^{(i)})x^{(i)}$$
batch examples single weight:  
$$\theta_j := \theta_j - \alpha\sum_{i=0}^n(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$$  
batch examples all weights (__BGD__):  
$$\theta := \theta - \alpha\sum_{i=0}^n(h_\theta(x^{(i)})-y^{(i)})x^{(i)}$$

#### Normal equation  
$$\theta = (X^TX)^{-1}X^Ty$$