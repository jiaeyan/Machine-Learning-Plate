#### Model  

1. Entropy:  
Uncertainty of a random variable.  
$$H(X) = -\sum_{x \in X}p(x)log(p(x))$$
When $X$ distribution is uniform, $H(X)$ is the max, very uncertain, could be anything. The model with max entropy is the best statistical model among all possible models.  

2. Conditional Entropy:  
Known the distribution of X, the expected uncertainty of Y.  
$$\begin{aligned}
H(Y|X) &= \sum_{x \in X}p(x)H(Y|X=x) \\
       &= -\sum_{x \in X}p(x)\sum_{y \in Y}p(y|x)log(p(y|x)) \\
       &= -\sum_{x \in X}\sum_{y \in Y}p(x)p(y|x)log(p(y|x)) \\
       &= -\sum_{x \in X}\sum_{y \in Y}p(x, y)log(p(y|x))
\end{aligned}$$

3. Maximum Entroy:  
   Under known constraints, assign uniform distribution to unkown facts to maximize entropy.
   __Constraints__:
   * the feature distributions of the empirical expectations should equal to the model expectations
     * $f(X=x, Y=y) = \begin{cases} 1, \quad if X=x, Y=y\\0, \quad otherwise \end{cases}$
     * from data, we get:
         * $\tilde{p}(X=x, Y=y) = \frac {C(X=x, Y=y)} {N}$
         * $\tilde{p}(X=x) = \frac {C(X=x)} {N}$
     * empirical expecations:
       * $E_{emp}(f_k) = \sum_{x \in X,y \in Y}\tilde{p}(x,y)f_k(x,y)$
     * model expectations:
       * $E_{model}(f_k) = \sum_{x \in X, y \in Y}\tilde{p}(x)p(y|x)f_k(x,y)$
     * $E_{emp}(f_k) = E_{model}(f_k) \rightleftharpoons E_{emp}(f_k) - E_{model}(f_k) = 0$
     * $\sum_{x \in X, y \in Y}\tilde{p}(x,y)f_k(x,y) = \sum_{x \in X, y \in Y}\tilde{p}(x)p(y|x)f_k(x,y)$
     * for all features 0-k:
       * $\sum_{k \in K, x \in X, y \in Y}\tilde{p}(x,y)f_k(x,y) = \sum_{k \in K, x \in X, y \in Y}\tilde{p}(x)p(y|x)f_k(x,y)$
   * $p(y|x) = argmax_pH(Y|X) = -\sum_{x \in X, y \in Y}\tilde{p}(x)p(y|x)log(p(y|x))$
   * $p(y|x) = argmin_p-H(Y|X) = \sum_{x \in X, y \in Y}\tilde{p}(x)p(y|x)log(p(y|x))$
   * $\sum_{y \in Y}p(y|x) = 1 \rightleftharpoons 1 - \sum_{y \in Y}p(y|x) = 0$  
  
4. How to compute $p(y|x)$? __Lagrange multipliers__ _(tool for constraint optimization)_!
   $$\begin{aligned}
   Lag(p, w) &= -H(p) + \lambda_0(1-\sum_{y \in Y}p(y|x)) + \sum_{k \in K}\lambda_k(E_{emp}(f_k) - E_{model}(f_k)) \\
             &= \sum_{x \in X, y \in Y}\tilde{p}(x)p(y|x)log(p(y|x)) + \lambda_0(1-\sum_{y \in Y}p(y|x)) + \sum_{k \in K}\lambda_k(\sum_{x \in X,y \in Y}\tilde{p}(x,y)f_k(x,y) - \sum_{x \in X, y \in Y}\tilde{p}(x)p(y|x)f_k(x,y))\end{aligned}$$
   $$p(y|x) = \frac {e^{\sum_{k \in K}\lambda_kf_k(x,y)}} {\sum_{y \in Y}e^{\sum_{k \in K}\lambda_kf_k(x,y)}}$$  

#### Likelihood  
$$\begin{aligned}
L(\theta) &= log\prod_{i=0}^np(y^{(i)}|x^{(i)})\\
          &= \sum_{i=0}^nlog(p(y^{(i)}|x^{(i)})) \\
          &= \sum_{i=0}^nlog(\frac {e^{\sum_{k \in K}\lambda_kf_k(x^{(i)},y^{(i)})}} {\sum_{y \in Y}e^{\sum_{k \in K}\lambda_kf_k(x^{(i)},y)}}) \\
          &= \sum_{i=0}^n\sum_{k \in K}\lambda_kf_k(x^{(i)},y^{(i)}) - \sum_{i=0}^nlog\sum_{y \in Y}e^{{\sum_{k \in K}\lambda_kf_k(x^{(i)},y)}}
\end{aligned}$$  

#### Loss function  
The negative log likelihood.  

#### Update rule  
$\sum_{i=0}^nf_k(x^{(i)}, y^{(i)}) - \sum_{i=0}^n\sum_{y \in Y}p(x^{(i)}, y)f_k(x^{(i)}, y) = 0$
which is exactly $E_{emp}(f_k) - E_{model}(f_k) = 0$

$\begin{aligned}
\theta &:= \theta + \alpha(E_{emp}(f_k) - E_{model}(f_k)) \\
        &= \theta + \alpha(\sum_{i=0}^nf_k(x^{(i)}, y^{(i)}) - \sum_{i=0}^n\sum_{y \in Y}p(x^{(i)}, y)f_k(x^{(i)}, y))
\end{aligned}$
