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
$S$: the possible hidden state sequence  
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

Besides to the regular probability matrix like forward algorith, we need another __backpointer matrix__ that records the paths.  

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