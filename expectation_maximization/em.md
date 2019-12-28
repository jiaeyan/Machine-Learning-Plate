### Main idea
EM could be used for unsupervised learning for generative models.  
Given seen observations ($Y$, _incomplete data_) of __one certain event__ ($Y$ value is fixed, rather than a set), maximize __the likelihood of this particular event__ by enumerating all hidden states ($Z$):  
$$\begin{aligned}
L(\theta) &= logP(Y|\theta) \\ 
          &= log\sum_Z(Y, Z|\theta) \\
          &= log\left( \sum_ZP(Y|Z, \theta)P(Z|\theta) \right)
\end{aligned}$$
However there is __no closed form solution__, and this likelihood is hard to maximize because
* hidden variable $Z$ is unknown
* there is log of sum/product  

Instead of resolving $\theta$ all at once, we maximize it by iteration of updating $\theta$, so $L(\theta) > L(\theta^{(i)})$:  
$$\begin{aligned}
L(\theta) - L(\theta^{(i}) &= log\left( \sum_ZP(Y|Z, \theta)P(Z|\theta) \right) - logP(Y|\theta^{(i)}) \\
&= \textcolor{blue}{log}\left( \textcolor{blue}{\sum_Z} \textcolor{green} {P(Z|Y,\theta^{(i)})} \frac {P(Y|Z, \theta)P(Z|\theta)} {\textcolor{green} {P(Z|Y, \theta^{(i)})}} \right) - logP(Y|\theta^{(i)}) \\
&\ge \textcolor{blue}{\sum_Z}\textcolor{green}{P(Z|Y, \theta^{(i)})} \textcolor{blue}{log} \frac {P(Y|Z, \theta)P(Z|\theta)} {\textcolor{green} {P(Z|Y, \theta^{(i)})}} - logP(Y|\theta^{(i)})\\
&= \textcolor{blue}{\sum_Z}\textcolor{green}{P(Z|Y, \theta^{(i)})} \textcolor{blue}{log} \frac {P(Y|Z, \theta)P(Z|\theta)} {\textcolor{green} {P(Z|Y, \theta^{(i)})}} - \textcolor{green}{\sum_ZP(Z|Y, \theta^{(i)})}logP(Y|\theta^{(i)}) \\
&= \sum_ZP(Z|Y, \theta^{(i)})\left[ log\frac {P(Y|Z, \theta)P(Z|\theta)} {P(Z|Y, \theta^{(i)})} - logP(Y|\theta^{(i)}) \right] \\
&= \sum_ZP(Z|Y, \theta^{(i)})\left[ log\frac {P(Y|Z, \theta)P(Z|\theta)} {P(Z|Y, \theta^{(i)})P(Y|\theta^{(i)})} \right]
\end{aligned}$$

* Blue colors are from __Jensen inequality__
* $\sum_ZP(Z|Y, \theta^{(i)}) = 1$, since $Y$ and $\theta^{(i)}$ are fixed, and this is just the distribution of $Z$, sum over as 1.

Then we define a helper function $B(\theta, \theta^{(i)})$:  
$$B(\theta, \theta^{(i)}) = L(\theta^{(i)}) + \sum_ZP(Z|Y, \theta^{(i)})\left[ log\frac {P(Y|Z, \theta)P(Z|\theta)} {P(Z|Y, \theta^{(i)})P(Y|\theta^{(i)})} \right]$$
So:  
$$L(\theta) \ge B(\theta, \theta^{(i)})$$
$$L(\theta^{(i)}) = B(\theta^{(i)}, \theta^{(i)})$$
So every $\theta^{(i)}$ that maximizes $B(\theta, \theta^{(i)})$ also maximizes $L(\theta^{(i)})$. When we choose $\theta^{(i+1)}$ to maximize $B(\theta, \theta^{(i)})$, we have:
$$\begin{aligned}
\theta^{(i+1)} &= \argmax_\theta B(\theta, \theta^{(i)}) \\
&= \argmax_\theta \left( \textcolor{red}{L(\theta^{(i)})} + \sum_ZP(Z|Y, \theta^{(i)}) log\frac {P(Y|Z, \theta)P(Z|\theta)} \textcolor{red}{{P(Z|Y, \theta^{(i)})P(Y|\theta^{(i)})}} \right) \\
&= \argmax_\theta \left( \sum_ZP(Z|Y, \theta^{(i)}) log P(Y|Z, \theta)P(Z|\theta) \right) \\
&= \argmax_\theta \left( \sum_ZP(Z|Y, \theta^{(i)}) log P(Y, Z|\theta) \right) \\
&= \argmax_\theta Q(\theta, \theta^{(i)})
\end{aligned}$$
* We can omit the red parts because they are fixed constant numbers (e.g., $\theta^{(i)}$) in terms of computing $\theta$.  

So far we get the $Q$ function, which means the expectation of the conditional probability distribution of unknown states $Z$, given observed $Y$ and current $\theta^{(i)}$:
$$\begin{aligned}
Q(\theta, \theta^{(i)}) &= \sum_ZP(Z|Y, \theta^{(i)})logP(Y, Z|\theta) \\
                        &= E_{Z|Y,\theta_n}[logP(Y,Z|\theta)]
\end{aligned}$$

### 3-coin problem
Each event/experiment is independent from each other. The likelihood of 1 event is $P(Y|\theta)$, for a set of $n$ events instead, we have:  
$$\begin{aligned}
L(\theta) &= \prod_{j=0}^nP(y_j|\theta) \\
          &= \sum_{j=0}^nlogP(y_j|\theta) \\
          &= \sum_{j=0}^nlog\sum_{z \in Z}P(y_j,z|\theta) \\
          &= \textcolor{blue}{\sum_{j=0}^n}log\sum_{z \in Z}P(y_j|z,\theta)P(z|\theta)
\end{aligned}$$

This looks the same as single event's likelihood, besides to the sum over all events' likelihoods. So we have the $Q$ function for this 3-coin problem as :
$$\begin{aligned}
Q(\theta, \theta^{(i)}) &= \textcolor{blue}{\sum_{j=0}^n}\sum_{z \in Z}\textcolor{blue}{\underline{P(z|y_j, \theta^{(i)})}}log \textcolor{blue}{\underline{P(y_j, z|\theta)}} \\
&= \sum_{j=0}^n\left[ P(z=1|y_j, \theta^{(i)})logP(y_j, z=1|\theta) + P(z=0|y_j, \theta^{(i)})logP(y_j,z=0|\theta)\right]
\end{aligned}$$

For 3-coin problem specifically, the probabilities of each component in the $Q$ function are as follow:  

First compute some necessary aux values:
$$\begin{aligned}
P(y_j,z|\theta^{(i)}) &= P(z|\theta^{(i)})P(y_j|z,\theta^{(i)}) \\ 
&= \begin{cases}
a^{(i)}(b^{(i)})^{y_j}(1-b^{(i)})^{1-y_j} &\text{if }z=1; \\
(1-a^{(i)})(c^{(i)})^{y_j}(1-c^{(i)})^{1-y_j} &\text{if }z=0.
\end{cases} &
\end{aligned}$$

Note $a^{(i)}, b^{(i)}, c^{(i)}$ here are known values from $\theta^{(i)}$.

$$\begin{aligned} P(y_j|\theta^{(i)}) 
&= \sum_{z \in Z}P(y_j,z|\theta^{(i)})\\
&=\sum_{z \in Z}P(z|\theta^{(i)})P(y_j|z,\theta^{(i)})\\
&=P(z=1|\theta^{(i)})P(y_j|z=1,\theta^{(i)})+P(z=0|\theta^{(i)})P(y_j|z=0,\theta^{(i)}) \\
&=\begin{cases} a^{(i)}b^{(i)}+(1-a^{(i)})c^{(i)}, & \text{if }y_j=1;\\ a^{(i)}(1-b^{(i)})+(1-a^{(i)})(1-c^{(i)}), & \text{if }y_j=0. \end{cases} \\
&=a^{(i)}(b^{(i)})^{y_j}(1-b^{(i)})^{1-y_j}+(1-a^{(i)})(c^{(i)})^{y_j}(1-c^{(i)})^{1-y_j} \end{aligned}$$

Then comes to the key components:

$$\begin{aligned}
\textcolor{blue}{\underline{P(z|y_j,\theta^{(i)})}} &= \frac {P(y_j, z|\theta^{(i)})} {P(y_j|\theta^{(i)})} \\
&= \begin{cases} \mu_j^{(i)} &= \frac{a^{(i)}(b^{(i)})^{y_j}(1-b^{(i)})^{1-y_j}}{a^{(i)}(b^{(i)})^{y_j}(1-b^{(i)})^{1-y_j}+(1-a^{(i)})(c^{(i)})^{y_j}(1-c^{(i)})^{1-y_j}} & \text{if }z=1; 
\\1-\mu_j^{(i)} &= \frac{(1-a^{(i)})(c^{(i)})^{y_j}(1-c^{(i)})^{1-y_j}}{a^{(i)}(b^{(i)})^{y_j}(1-b^{(i)})^{1-y_j}+(1-a^{(i)})(c^{(i)})^{y_j}(1-c^{(i)})^{1-y_j}} & \text{if }z=0.
\end{cases}
\end{aligned}$$

$$\begin{aligned}
\textcolor{blue}{\underline{P(y_j,z|\theta)}} &= P(z|\theta)P(y_j|z,\theta) \\ 
&= \begin{cases}
ab^{y_j}(1-b)^{1-y_j} &\text{if }z=1; \\
(1-a)c^{y_j}(1-c)^{1-y_j} &\text{if }z=0.
\end{cases} &
\end{aligned}$$

Note $a, b, c$ here are unkown parameters from $\theta$, the target that we are going to compute in the task.  

So now we get the $Q$ function as:  

$$\begin{aligned}
Q(\theta, \theta^{(i)}) &= \sum_{j=0}^n\sum_{z \in Z} P(z|y_j, \theta^{(i)}) log P(y_j, z|\theta) \\
&= \sum_{j=0}^n\left[ P(z=1|y_j, \theta^{(i)})logP(y_j, z=1|\theta) + P(z=0|y_j, \theta^{(i)})logP(y_j,z=0|\theta)\right] \\
& = \sum_{j=0}^n \left[ \mu_j^{(i)} \times ab^{y_j}(1-b)^{1-y_j} + (1 - \mu_j^{(i)}) \times (1-a)c^{y_j}(1-c)^{1-y_j} \right]
\end{aligned}$$

So for E step, the key point is to compute $\mu_j^{(i)}$.

#### E-step
$$\mu_j^{(i+1)} = \frac {a^{(i)}(b^{(i)})^{y_j}(1-b^{(i)})^{1-y_j}} {a^{(i)}(b^{(i)})^{y_j}(1-b^{(i)})^{1-y_j} + (1-a^{(i)})(c^{(i)})^{y_j}(1-c^{(i)})^{1-y_j}}$$

#### M-step
Take derivation from $Q(\theta, \theta^{(i)})$ by each parameter in $\theta$, to compute the new $a^{(i+1)}, b^{(i+1)}, c^{(i+1)}$, then get the parameter update formula:

$$a^{(i+1)} = \frac 1 n \sum_{j=1}^n\mu_j^{(i+1)}$$
$$b^{(i+1)} = \frac {\sum_{j=1}^n\mu_j^{(i+1)}y_j} {\sum_{j=1}^n\mu_j^{(i+1)}}$$
$$c^{(i+1)} = \frac {\sum_{j=1}^n(1-\mu_j^{(i+1)})y_j} {\sum_{j=1}^n(1-\mu_j^{(i+1)})}$$