####Model  
$$h_\theta(x) = sign(z(x)) = sign(\theta^Tx)$$
$$z_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3 + ... = \sum_{i=0}^d\theta_ix_i = \theta^Tx$$ 
$$sign(x) = \begin{cases} 
+1, &x \geq 0 \\
-1, &x < 0
\end{cases}$$  

Assume there is a hyperinterface that makes $\theta^Tx = 0$.  

####Likelihood  
Since a class is directly predicted, rather than a probability, there is no likelihood for the data set.  

####Loss function  
Since the function of number of misclassified instances (M) is not differentiable, we use the __distance of misclassified instance to the hyperinterface__ instead:  
$$distance = \frac 1{\rVert w \lVert _2}\lvert\theta^Tx\rvert$$
where $\rVert w \lVert _2$ is $L_2$ norm of $w$, a scalar:
$$\rVert w \lVert _2 = \sqrt{\sum_{i=0}^nw_i^2}$$
For any misclassified instance, $y_i$ is always the oppsite of $\hat{y_i}$, so $y_i\hat y_i < 0$, since $\hat{y_i} = \theta^Tx_i$, we have:  
$$-y_i(\theta^Tx_i) = \lvert\theta^Tx\rvert > 0$$
So the distance from misclassified instance to hyperinterface is:  
$$distance = -\frac 1{\rVert w \lVert _2}y_i(\theta^Tx_i)$$
So we have loss:  
$$L(\theta) = -\sum_{x_i \in M}\frac 1{\rVert w \lVert _2}y_i(\theta^Tx_i)$$
Since ${\rVert w \lVert _2}$ is the same for all misclassified instances, we can omit it and get:  
$$L(\theta) = -\sum_{x_i \in M}y_i(\theta^Tx_i) \ge0$$

####Update rule  
Collect all misclassified instances, then update 1 by 1 (only __SGD__):  
$$\theta := \theta - (-\alpha y_ix_i) = \theta + \alpha y_ix_i$$