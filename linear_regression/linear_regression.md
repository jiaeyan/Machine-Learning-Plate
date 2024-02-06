### Model  
__Feature representation__: linear regression computes a real number value given an instance $x$ with a set of features and a set of parameters $\theta$ corresponding to each feature.  

$$h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3 + ... = \sum_{i=0}^d\theta_ix_i = \theta^Tx$$   

$\theta_0$ is generally referred as the bias/b.  

The linear regression model can be interpreted as a *__simple neural network__*:  
- it has a real-valued weight vector  
- it has a real-valued bias  
- it uses the _identity function_ as its activation function _(the result itself)_  

### Assumptions  
- The dataset is produced by some unknown function which maps features to labels  

### Likelihood  
Since a real number is predicted, rather than a probability, there is no likelihood for the data set.  

Different from Naive Bayes Classifier, a generative model, this is a discriminative model who computes target conditional probablity without likelihood and prior.

### Loss function  
Also known as cost function.  
*__Least Mean Square Error (MSE)__* cost function: the average squared difference between the estimated values and the actual values.  
This is what we want to minimize. We want to find all $\theta$ that minimize this; in other words, the optimal/minimum point of the convex (boat shape).  

In regression, local optimal = global optiaml.  

$$\begin{aligned}
J(\theta) &= \frac 1{2n}\sum_{i=0}^n(\hat{y}^{(i)}-y^{(i)})^2 \\
          &= \frac 1{2n}\sum_{i=0}^n(h_\theta(x^{(i)})-y^{(i)})^2
\end{aligned}$$  

### Train with Gradient Descent: updating rule  
Use the partial derivative of the cost function to minimize the cost value with respect to each parameter.  
If the derivative/切线斜率 > 0 on the convex, means $\theta$ could be smaller (move left to the convex to get closer to the optimal), thus minus; if < 0, means $\theta$ should be bigger (move right to the convex to get closer to the optimal), thus add; always in opposite conditions, thus minus the derivative.  

Derivative of single parameter of a single example = $(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$  
Derivative of single parameter of batch examples = $\frac1n(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$  

single example single weight updating:  
$$\theta_j := \theta_j - \alpha(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$$  
single example all weights updateing (__SGD__):  
$$\theta := \theta - \alpha(h_\theta(x^{(i)})-y^{(i)})x^{(i)}$$
batch examples single weight updating:  
$$\theta_j := \theta_j - \alpha\frac1n\sum_{i=0}^n(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$$  
batch examples all weights updating (__BGD__):  
$$\theta := \theta - \alpha\frac1n\sum_{i=0}^n(h_\theta(x^{(i)})-y^{(i)})x^{(i)}$$

### Train with Normal Equation  
Closed form solution:
$$\theta = (X^TX)^{-1}X^Ty$$