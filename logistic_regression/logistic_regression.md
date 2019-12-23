#### Model  
Logistic regression computes a __probability__ _(different from linear regression)_ of the most possible class for an instance.  
Though its output is still continuous like linear regression, the classfication result is on top of the continuous real number, e.g., take the probability of 0.5 as a threshold between binary classes.  

$$h_\theta(x) = \frac 1 {1 + e^{-z(x)}} = \frac 1 {1 + e^{-\theta^Tx}}$$
$$z_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3 + ... = \sum_{i=0}^d\theta_ix_i = \theta^Tx$$   

#### Likelihood  
Assume:  
$p(y=1|x;\theta) = h_\theta(x)$
$p(y=0|x;\theta) = 1 - h_\theta(x)$
$p(y|x;\theta) = (h_\theta(x))^y(1-h_\theta(x))^{1-y}$

Likelihood is the product of probabilities of all data (__MLE__, maximum likelihood estimation):  
$\begin{aligned}
L(\theta) &= p(y|X;\theta) \\
          &= \prod_{i=0}^np(y^{(i)}|x^{(i)};\theta) \\
          &= \prod_{i=0}^n(h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}}
\end{aligned}$

To make things easier, take log:  
$logL(\theta) = \sum_{i=0}^ny^{(i)}logh_\theta(x^{(i)}) + (1-y^{(i)})log(1-h_\theta(x^{(i)}))$

#### Loss function  
Cross entropy (loss) between real class and predicted class (two Bernoulii Distributions) for 1 instance:  
$$H(y, \hat{y}) = -\sum_{i=0}^ny_ilog\hat{y}_i = -ylog\hat{y} - (1-y)log(1-\hat{y})$$
So we get:  
$$-H(y, \hat{y}) = \sum_{i=0}^ny_ilog\hat{y}_i = ylog\hat{y} + (1-y)log(1-\hat{y})$$
which looks exactly the same as the log likelihood of 1 instance above. So to compute the loss of the whole data, we can compute the sum of __negative log likelihood__ of the whole data set.  
Cross entropy between real class and predicted class for all instances:  
$$J(\theta) = \sum_{i=0}^n-H(y, \hat{y})= -logL(\theta) = -(\sum_{i=0}^ny^{(i)}logh_\theta(x^{(i)}) + (1-y^{(i)})log(1-h_\theta(x^{(i)})))$$

#### Update rule  
single example single weight:  
$$\theta_j := \theta_j - \alpha(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$$  
single example all weights (__SGD__):  
$$\theta := \theta - \alpha(h_\theta(x^{(i)})-y^{(i)})x^{(i)}$$
batch examples single weight:  
$$\theta_j := \theta_j - \alpha\sum_{i=0}^n(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$$  
batch examples all weights (__BGD__):  
$$\theta := \theta - \alpha\sum_{i=0}^n(h_\theta(x^{(i)})-y^{(i)})x^{(i)}$$