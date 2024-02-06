# Models  
## 1. Classification  
### - Generative models
- __Native Bayes Classifier__: Maximum Likelihood Estimation, models uncertainties  
  $\begin{aligned}
    \overbrace{\hat{y}}^{posterior} &= \argmax_{y \in Y}P(y|x) \\
                                    &= \argmax_{y \in Y}\frac {P(x|y)P(y)} {P(x)} \\
                                    &= \argmax_{y \in Y}\overbrace{P(x|y)}^{likelihood}\overbrace{P(y)}^{prior} \\
                                    &= \argmax_{y \in Y}\overbrace{P(x_1x_2x_3\dots x_n|y)}^{likelihood}\overbrace{P(y)}^{prior} \\
                                    &= \argmax_{y \in Y}\overbrace{P(x_1|y)P(x_2|y)P(x_3|y)\dots P(x_n|y)}^{likelihood}\overbrace{P(y)}^{prior} \\
                                    &= \argmax_{y \in Y}\overbrace{\prod_{i=0}^nP(x_i|y )}^{likelihood}\overbrace{P(y)}^{prior} \\
                                    &= \argmax_{y \in Y}\overbrace{\sum_{i=0}^nlog(P(x_i|y))}^{likelihood} + \overbrace{log(P(y))}^{prior}  
    \end{aligned}$
### - Discriminative models
- __Linear Regression__: outputs a continuous value, $y\in\{-\infty, +\infty\}$; a simple neural network with identity activation function  
  - Feature representation / Regression function  
    - $h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3 + ... = \sum_{i=0}^d\theta_ix_i = \theta^Tx$  
  - Loss function: Least Mean Square Error  
    - $\begin{aligned}J(\theta) &= \frac 1{2n}\sum_{i=0}^n(\hat{y}^{(i)}-y^{(i)})^2 \\&= \frac 1{2n}\sum_{i=0}^n(h_\theta(x^{(i)})-y^{(i)})^2\end{aligned}$  
  - Gradient Descent: weight updating rule, $\alpha$ as learning rate  
    - $\theta := \theta - \alpha\frac1n\sum_{i=0}^n(h_\theta(x^{(i)})-y^{(i)})x^{(i)}$
- __Logistic Regression__: outputs a discrete value, $y\in[0,1]$; a simple neural network with sigmoid activation function  
  - Feature representation  
    - $z_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3 + ... = \sum_{i=0}^d\theta_ix_i = \theta^Tx$ 
  - Classification function: converts $\{-\infty, +\infty\}$ to $[0,1]$  
    - Sigmoid function: $h_\theta(x) = \frac 1 {1 + e^{-z(x)}} = \frac 1 {1 + e^{-\theta^Tx}}$  
  - Loss function: Cross Entropy (Maximum Likelihood Estimation)  
    - $J(\theta) = -\frac1n\sum_{i=0}^n\left[y^{(i)}logh_\theta(x^{(i)}) + (1-y^{(i)})log(1-h_\theta(x^{(i)}))\right]$  
  - Gradient Descent: weight updating rule, $\alpha$ as learning rate  
    - $\theta := \theta - \alpha\frac1n\sum_{i=0}^n(h_\theta(x^{(i)})-y^{(i)})x^{(i)}$

# Evaluation  
## 1. Classification
Confusion matrix:  

|   | gold positive  |  gold negative |   |   |
|:---:|:---:|:---:|:---:|:---:|
|  system positive |  true positive  |  false positive <br> _(type 1 error)_ | $precision=\frac {tp} {tp+fp}$|
|  system negative | false negative <br> _(type 2 error)_  |  true negative |   |   |
|   |  $recall=\frac {tp} {tp+fn}$ |   |  $accuracy=\frac {tp+tn} {tp+fp+tn+fn}$ | $F_1=\frac {2PR} {P+R}$  |

- _P/Precision_: in terms of all the system positives, how many of them are gold positives;  
- _R/Recall_: Sensitivity, in terms of all the gold positives, how many of them are system positives;  
- $F_1$: comes from a weighted harmonic mean of precision and recall  
