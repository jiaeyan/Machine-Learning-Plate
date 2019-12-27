#### Main idea  
$$\hat{y} = \argmax_{y \in Y}P(y|x) = \argmax_{y \in Y}\frac {P(x|y)P(y)} {P(x)}$$
Since $P(x)$ is the same for all labels, omit it:  
$$\hat{y} = \argmax_{y \in Y}P(y|x) = \argmax_{y \in Y}\overbrace{P(x|y)}^{likelihood}\overbrace{P(y)}^{prior}$$
According to the feature independence assumption:  
$$\begin{aligned}
\hat{y} &= \argmax_{y \in Y}P(y|x) \\ 
        &= \argmax_{y \in Y}\overbrace{P(x|y)}^{likelihood}\overbrace{P(y)}^{prior} \\
        &= \argmax_{y \in Y}\overbrace{P(x_1x_2x_3\dots x_n|y)}^{likelihood}\overbrace{P(y)}^{prior} \\
        &= \argmax_{y \in Y}\overbrace{P(x_1|y)P(x_2|y)P(x_3|y)\dots P(x_n|y)}^{likelihood}\overbrace{P(y)}^{prior} \\
        &= \argmax_{y \in Y}\overbrace{\prod_{i=0}^nP(x_i|y )}^{likelihood}\overbrace{P(y)}^{prior}
\end{aligned}$$
As usual, we take log on the probability:
$$\hat{y} = \argmax_{y \in Y}\overbrace{\sum_{i=0}^nlog(P(x_i|y))}^{likelihood} + \overbrace{log(P(y))}^{prior}$$

#### Multinomial
$P(x_i|y) = \frac {count(x_i, y) + 1} {\sum_{x_i \in V}(count(x_i, y) + 1)} = \frac {count(x_i, y) + 1} {\sum_{x_i \in V}(count(x_i, y)) + |V|}$
$P(y) = \frac {\sum_{x_i \in V}count(x_i, y)} {\sum_{x_i \in V}count(x_i)}$
$P(y|x) = P(y)\prod_{i=0}^nP(x_i|y)$
n = number of observed features in x as true evidence

#### Bernoulli
$P(x_i|y) = \frac {N_{x_i \wedge y} + 1} {N_y + |Y|}$
$P(y) = \frac {N_y} {\sum_{y \in Y}N_y}$
$P(y|x) = P(y)\prod_{i=0}^nP(x_i|y)\prod_{j=0}^m(1-P(x_j|y))$
n = number of observed features in x as true evidence  
m = number of unobserved features in x as false evidence  
n + m = total number of features one instance could have in space

#### Gaussian
Multinomial and Bernoulli models only work for discrete categorical features, for continuous features, we use Gaussian NB.  
For each feature dimension, compute its mean and standard deviation：
$\mu = \frac {\sum_{i=0}^nx_i} {n}$

$\sigma = \sqrt{\frac {\sum_{i=0}^n(x_i - \mu)^2} {n}}$

Each label has different means and standard deviations for each feature dimension.  
$P(x_i|y) = \frac {1}{\sigma_{iy} \sqrt{2\pi}}e^{-\frac {(x_i-\mu_{iy})^2}{2\sigma_{iy}^2}}$
$P(y) = \frac {N_y} {\sum_{y \in Y}N_y}$
$P(y|x) = P(y)\prod_{i=0}^nP(x_i|y)$

Given an instance, we simply plug in its feature values to the formula above to predict its label.