#### Assumptions
1. (1st order) __Markov assumption__ (for transition probability):
   $p(s_i|s_1, s_2, s_3, \dots, s_{i-1}) = p(s_i|s_{i-1})$
2. __Output independence__ (for emission probability)
   $p(o_i|s_1, s_2, s_3, \dots, s_{i-1}) = p(o_i|s_i)$

#### Components
* $S = \{s_1, s_2, s_3, \dots, s_i\}$: a set of hidden states
* $O = \{o_1, o_2, o_3, \dots, o_i\}$: a set of observed events
* $A = a_{11}\dots a_{ij} \dots a_{NN}$: the state transition probability matrix
* $B = b_i(o_t)$: the emission probability matrix
* $\pi = \{\pi_1, \pi_2, \pi_3, \dots, \pi_i\}$: the initial state probability distribution

#### Likelihood
With HMM $\lambda = (A, B, \pi)$, compute the likelihood of observation sequence $O$.  

##### 1. Naive approach
$O$: the observed sequence  
$S$: one possible hidden state sequence  
So we get the joint probability of them:  
$$P(O, S) = P(O|S)P(S) = \prod_{i=1}^TP(o_i|s_i) \times \prod_{i=1}^TP(s_i|s_{i-1})\times \pi_{s_0}$$
Now we can compute the probability of the observed sequence $O$ with all possible hidden state sequences by marginal probability:
$$\begin{aligned}
P(O) &= \sum_SP(O,S) \\
     &= \sum_SP(O|S)P(S) \\
     &= \sum_S\prod_{i=1}^TP(o_i|s_i) \times \prod_{i=1}^TP(s_i|s_{i-1}) \times \pi_{s_0}
\end{aligned}$$  
However, time complexity is huge $TN^T$. Instead we use forward algorithm, a dynamic programming approach, whose time complexity is $TN^2$.  
##### 2. Forward algorithm
Define the probability of being in state $j$ after seeing $t$ observations (__with all possible hidden state sequences till $\bold t$__), given the parameter sets $\lambda(A, B, \pi)$
$$\begin{aligned}
\alpha_t(j) &= P(o_1, o_2, o_3, \dots, o_t, s_t=j) \\
            &= \left[ \sum_{i=1}^N\alpha_{t-1}(i)a_{ij}\right] b_j(o_t)
\end{aligned}$$
* __Initialization__
  $$\alpha_1(j) = \pi_jb_j(o_1) \quad 1 \le j \le N$$
* __Loop__
  $$\alpha_t(j) = \left[ \sum_{i=1}^N\alpha_{t-1}(i)a_{ij}\right] b_j(o_t) \quad 1 \le j \le N, 1 < t \le T$$
* __Termination__
  $$P(O|\lambda) = \sum_{i=1}^N\alpha_T(i)$$

##### 3. Backward algorithm
Define the probability of seeing the observations from time $t + 1$ to $T$ (__with all possible hidden state sequences till $\bold{t+1}$__), given
that we are in state $j$ at time $t$.  
$$\beta_t(j) = P(o_{t+1}, o_{t+2}, \dots, o_T|s_t=j)$$
* __Initialization__
  $$\beta_T(j) = 1 \quad 1 \le j \le N$$
* __Loop__
  $$\beta_t(j) = \sum_{i=1}^N\beta_{t+1}(i)a_{ji}b_i(o_{t+1}) \quad 1 \le j \le N, 1 \le t < T$$
* __Termination__
  $$P(O|\lambda) = \sum_{i=1}^N\pi_i\beta_1(i)b_i(o_1)$$

#### Decoding
With HMM $\lambda = (A, B, \pi)$, and the  observation sequence $O$, compute the most possible hidden state sequence $S$. 
##### 1. Naive approach
Enumerate all possible hidden state sequences, compute relative likelihoods with forward algorithm, and choose the biggest one. Still with time complexity of $N^T$ to list all state sequences.
##### 2. Viterbi algorithm
Define the probability of being in state $j$ after seeing $t$ observations and passing through the most possible state sequence $s_1 \dots s_{t−1}$ that maximizes the likelihood.
$$v_t(j) = \max_{s_1, \dots, s_{t-1}}P(s_{1} \dots s_{t-1}, o_1 \dots o_t, s_t = j)$$  

Besides to the regular probability matrix like forward algorithm, we need another __backpointer matrix__ that records the paths, where each cell records the coming state id.  

* __Initialization__
  $$v_1(j) = \pi_jb_j(o_1)  \quad 1 \le j \le N$$
  $$bt_1(j) = 0  \quad 1 \le j \le N$$
* __Loop__
  $$v_t(j) = \max_{i=1}^Nv_{t-1}(i)a_{ij} b_j(o_t) \quad 1 \le j \le N, 1 < t \le T$$
  $$bt_t(j) = \argmax_{i=1}^Nv_{t-1}(i)a_{ij} b_j(o_t) \quad 1 \le j \le N, 1 < t \le T$$
* __Termination__
  $$\hat P = \max_{i=1}^Nv_T(i)$$
  $$\hat s_T = \argmax_{i=1}^Nv_T(i)$$  

#### Learning  
Given the observation sequences $\sum_{i=0}^nO$, and a set of labels $\sum_{j=0}^mS$， compute the HMM parameters $\lambda = (A, B, \pi)$.  
This could be resolved by EM algorithm.  
##### 1. Expectations  
1. Given observation $O$ and and the $\lambda = (A, B, \pi)$, we have the probability of being at state $s_i$ at time $t$ _(no transition)_:  
   $$\gamma_t(i) = P(i_t = s_i | O,\lambda) = \frac{P(i_t = s_i ,O|\lambda)}{P(O|\lambda)} = \frac{P(i_t = s_i ,O|\lambda)}{\sum\limits_{j=0}^NP(i_t = s_j ,O|\lambda)}$$  
   By definitions of forward and backward algorithms:
   $$P(i_t = s_i ,O|\lambda) = \alpha_t(i)\beta_t(i)$$  
   Why?  
   $$\alpha_t(i) = P(o_1,o_2,...o_t, i_t =s_i | \lambda)$$  
   $$\beta_t(i) = P(o_{t+1},o_{t+2},...o_T| i_t =s_i, \lambda)$$  
   $$\begin{aligned}
   \alpha_t(i)\beta_t(i) &= P(o_1,o_2,...o_t, i_t =s_i | \lambda) \times P(o_{t+1},o_{t+2},...o_T| i_t =s_i , \lambda) \\
   &= P(i_t = s_i, o_1,o_2,...o_t, o_{t+1},o_{t+2},...o_T |\lambda) \\
   &= P(i_t = s_i, O|\lambda)
   \end{aligned}$$  
   This is probability of being at state $i$ at time $t$; we can sum over all probabilities of being at all states at this $t$, so we have:
   $$\gamma_t(i) = \frac {\alpha_t(i)\beta_t(i)}{\sum\limits_{j=1}^N \alpha_t(j)\beta_t(j)}$$
2. Given observation $O$ and and the $\lambda = (A, B, \pi)$, we have the probability of being at state $s_i$ at time $t$, and at state $s_j$ at time $t + 1$ _(transition)_:
   $$\xi_t(i,j) = P(i_t = s_i, i_{t+1}=s_j| O,\lambda) = \frac{P(i_t = s_i, i_{t+1}=s_j, O|\lambda)}{P(O|\lambda)} =\frac{P(i_t = s_i, i_{t+1}=s_j, O|\lambda)}{\sum\limits_{k=0}^N\sum\limits_{l=0}^NP(i_t = s_k, i_{t+1}=s_l, O|\lambda)}$$  
   By definitions of forward and backward algorithms:  
   $$P(i_t = s_i, i_{t+1}=s_j, O|\lambda) = \alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)$$  
   Why?  
   $$\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j) = P(o_1,o_2,...o_t, i_t =s_i | \lambda) \times a_{ij} \times b_j(o_{t+1}) \times P(o_{t+2},o_{t+3},...o_T| i_t =s_j, \lambda)$$  
   We can sum over all possible state transitions at this time $t$, so we have:  
   $$\xi_t(i,j) = \frac{\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}{\sum\limits_{k=0}^N\sum\limits_{l=0}^N\alpha_t(k)a_{kl}b_l(o_{t+1})\beta_{t+1}(l)}$$  
3. In all, we have:  
   * Expectation of being at state $i$ given observation $O$: $\sum\limits_{t=1}^T\gamma_t(i)$  
   * Expectation of transiting from state $i$ given observation $O$: $\sum\limits_{t=1}^{T-1}\gamma_t(i)$
   * Expectation of transiting from state $i$ to $j$ given observation $O$: $\sum\limits_{t=1}^{T-1}\xi_t(i,j)$  

##### 2. Maximization  
$Q$ function of EM (Baum-Welch):  
$$\begin{aligned}
Q(\lambda, \overline{\lambda}) &= \sum_{I}P(I|O,\overline{\lambda})logP(O,I|\lambda) \\
&= \sum_{I} \frac {P(O, I|\overline{\lambda})}{\textcolor{red}{P(O|\overline{\lambda})}}logP(O,I|\lambda) \\
&= \sum_{I}P(O, I|\overline{\lambda})logP(O,I|\lambda)
\end{aligned}$$  
So the $\overline{\lambda}$ we want could be computed from:  
$$\overline{\lambda} = \argmax_{\lambda}\sum\limits_{I}P(O,I|\overline{\lambda})logP(O,I|\lambda)$$  
Since 
$$P(O,I|\lambda) = \prod_{d=1}^D\pi_{i_1^{(d)}}b_{i_1^{(d)}}(o_1^{(d)})a_{i_1^{(d)}i_2^{(d)}}b_{i_2^{(d)}}(o_2^{(d)})...a_{i_{T-1}^{(d)}i_T^{(d)}}b_{i_T^{(d)}}(o_T^{(d)})$$  
Plug in and we get:  
$$\overline{\lambda} = \argmax_{\lambda}\sum\limits_{d=1}^D\sum\limits_{I}P(O,I|\overline{\lambda}) \left[ log\pi_{i_1} + \sum\limits_{t=1}^{T-1}log\;a_{i_t,i_{t+1}} + \sum\limits_{t=1}^Tlog b_{i_t}(o_t) \right]$$  
We could maximize 3 components 1 by 1:  
* $$\overline{\pi_i} = \argmax_{\pi_{i_1}} \sum\limits_{d=1}^D\sum\limits_{I}P(O,I|\overline{\lambda})log\pi_{i_1} = \argmax_{\pi_{i}} \sum\limits_{d=1}^D\sum\limits_{i=1}^NP(O,i_1^{(d)} =i|\overline{\lambda})log\pi_{i}$$  
  Since $\sum\limits_{i=1}^N\pi_i =1$ as a constraint, we apply Lagrange Multiplier Method here:  
  $$\overline{\pi_i} = \argmax_{\pi_{i}}\sum\limits_{d=1}^D\sum\limits_{i=1}^NP(O,i_1^{(d)} =i|\overline{\lambda})log\pi_{i} + \gamma(\sum\limits_{i=1}^N\pi_i -1)$$  
  Take derivation on above and make it equal to 0, we have:  
  $$\sum\limits_{d=1}^DP(O,i_1^{(d)} =i|\overline{\lambda}) + \gamma\pi_i = 0$$  
  Make $i$ from 1 to $N$ and sum them all over, we have:  
  $$\sum\limits_{d=1}^D\sum\limits_{n=1}^NP(O,i_1^{(d)} = n|\overline{\lambda}) + \sum\limits_{n=1}^N\gamma\pi_n = 0$$  
  Again since $\sum\limits_{i=1}^N\pi_i =1$, and $\sum\limits_{n=1}^NP(O,i_1 = n|\overline{\lambda}) = P(O|\overline{\lambda})$ _(marginal probability of $O$)_. Plug in these two we have:  
  $$\begin{aligned}
  \sum\limits_{d=1}^D\sum\limits_{n=1}^NP(O,i_1^{(d)} = n|\overline{\lambda}) + \gamma &= 0\\
  \sum\limits_{d=1}^DP(O|\overline{\lambda}) + \gamma &= 0
  \end{aligned}$$  
  Now we have value of $\gamma$, plug in $\gamma$ we have:  
  $$\pi_i =\frac{\sum\limits_{d=1}^DP(O,i_1^{(d)} =i|\overline{\lambda})}{\sum\limits_{d=1}^DP(O|\overline{\lambda})} = \frac{\sum\limits_{d=1}^DP(O,i_1^{(d)} =i|\overline{\lambda})}{DP(O|\overline{\lambda})} = \frac{\sum\limits_{d=1}^DP(i_1^{(d)} =i|O, \overline{\lambda})}{D} = \frac{\sum\limits_{d=1}^DP(i_1^{(d)} =i|O^{(d)}, \overline{\lambda})}{D}$$  
  Since in the Expectation step we have:  
  $$\gamma_1^{(d)}(i) = P(i_1^{(d)} =i|O^{(d)}, \overline{\lambda})$$  
  So we have:  
  $$\pi_i = \frac{\sum\limits_{d=1}^D\gamma_1^{(d)}(i)}{D}$$
* $$\overline{a_{ij}} = \argmax_{a_{ij}}\sum\limits_{d=1}^D\sum\limits_{I}\sum\limits_{t=1}^{T-1}P(O,I|\overline{\lambda})log\;a_{i_t,i_{t+1}} = \sum\limits_{d=1}^D\sum\limits_{i=1}^N\sum\limits_{j=1}^N\sum\limits_{t=1}^{T-1}P(O,i_t^{(d)} = i, i_{t+1}^{(d)} = j|\overline{\lambda})loga_{ij}$$  
  We have $\sum\limits_{j=1}^Na_{ij} =1$, use Lagrange Multiplier Method and take derivation and get:  
  $$\begin{aligned}
  a_{ij} &= \frac{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T-1}P(O^{(d)}, i_t^{(d)} = i, i_{t+1}^{(d)} = j|\overline{\lambda})}{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T-1}P(O^{(d)}, i_t^{(d)} = i|\overline{\lambda})} \\
  &= \frac {\frac {\sum\limits_{d=1}^D\sum\limits_{t=1}^{T-1}P(O^{(d)}, i_t^{(d)} = i, i_{t+1}^{(d)} = j|\overline{\lambda})}{\textcolor{green}{\sum\limits_{d=1}^DP(O^{(d)}|\overline{\lambda}))}}} {
      \frac 
      {\sum\limits_{d=1}^D\sum\limits_{t=1}^{T-1}P(O^{(d)}, i_t^{(d)} = i|\overline{\lambda})}
      {\textcolor{green}{\sum\limits_{d=1}^DP(O^{(d)}|\overline{\lambda}))}}
  } \\
  &= \frac 
  {\sum\limits_{d=1}^D\sum\limits_{t=1}^{T-1}P(i_t^{(d)} = i, i_{t+1}^{(d)} = j|O^{(d)}, \overline{\lambda})} 
  {\sum\limits_{d=1}^D\sum\limits_{t=1}^{T-1}P(i_t^{(d)} = i | O,\overline{\lambda})}
  \end{aligned}$$  
  Again, since in Expectation step, we have:  
  $$\xi_t^{(d)}(i,j) = P(i_t^{(d)} = i, i_{t+1}^{(d)} = j|O^{(d)}, \overline{\lambda})$$  
  $$\gamma_t^{(d)}(i) = P(i_t^{(d)} = i | O,\overline{\lambda})$$  
  So we have:  
  $$a_{ij} = \frac{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T-1}\xi_t^{(d)}(i,j)}{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T-1}\gamma_t^{(d)}(i)}$$

* $$\overline{}\sum\limits_{d=1}^D\sum\limits_{I}\sum\limits_{t=1}^{T}P(O,I|\overline{\lambda})log\;b_{i_t}(o_t) = \sum\limits_{d=1}^D\sum\limits_{j=1}^N\sum\limits_{t=1}^{T}P(O,i_t^{(d)} = j|\overline{\lambda})log\;b_{j}(o_t)$$  
  Since $\sum\limits_{k=1}^Mb_{j}(k) =1$, we apply the same procedure as above, and have:  
  $$\begin{aligned}
  b_{j}(k) &= \frac{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T}P(O,i_t^{(d)} = j|\overline{\lambda})I(o_t^{(d)}=v_k)}{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T}P(O,i_t^{(d)} = j|\overline{\lambda})} \\
  &= \frac{\sum\limits_{d=1}^D\sum\limits_{t=1, o_t^{(d)}=v_k}^{T}\gamma_t^{(d)}(j)}{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T}\gamma_t^{(d)}(j)}
  \end{aligned}$$
* In all, we have:  
  $$\pi_i = \frac{\sum\limits_{d=1}^D\gamma_1^{(d)}(i)}{D}$$  
  $$a_{ij} = \frac{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T-1}\xi_t^{(d)}(i,j)}{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T-1}\gamma_t^{(d)}(i)}$$  
  $$b_{j}(k) = \frac{\sum\limits_{d=1}^D\sum\limits_{t=1, o_t^{(d)}=v_k}^{T}\gamma_t^{(d)}(j)}{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T}\gamma_t^{(d)}(j)}$$
