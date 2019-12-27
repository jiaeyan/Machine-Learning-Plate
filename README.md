# Machine Learning Plate

This is a plate of machine learning algorithm implementation tastes made of Python and Numpy. Enjoy!  

1. Linear Regression: BGD, SGD, MBGD
2. Logistic Regression: BGD, SGD, MBGD
3. MaxEnt (Maximum Entropy Method)
4. Perceptron
5. KNN _(K-Nearest Neighbor)_
6. Naive Bayes Classifier: Multinomial, Bernoulli, Gaussian
7. HMM _(Hidden Markov Model)_: Forward, Backward, Viterbi, Baum-Welch algorithms
8. EM _(Expectation Maximization)_: 3-coin problem
9. _more algorithms on the way..._  

## Highlights
* __Independence__: each algorithm file is independent, no class inheritance, no function overload, no shared util functions, even many of them look exactly the same. We break the thumb of rule of programming here, so you can clone this repo and just grab any single implementation you want and directly embed to your own codes!
* __Plainness__: we use very plain codes to illustrate each algorithm, and prefer to use more straightforward function calls than high level syntax sugars from 3rd party libs, which could be hard to get their purposes from names quickly.
* __Cheatsheet__: it is not enough to understand an algorithm merely through its impelmentation codes, since most of them are results from theories, e.g., the weight update formula. We add a math cheatsheet along with every algorithm so you could pick up the math quickly!

## Conventions
* X: a data matrix, shaped as (|num_x|, |num_features|) 
* Y: a label vector, corresponding to X 
* W: a weight matrix, shaped as (|num_labels|, |num_features|)
* Z: a sum over probabilities of all labels / a vector showing sums over weights dot x features
* X[i]: the ith data record
* Y[i]: the ith label record, corresponding to X[i]
* W[i]: the weight vector for label i
* x: a feature vector, x = X[i] 
* y: a label scalar, corresponding to x, y = Y[i]
* w: a feature weight vector
* z: a scalar showing sums over weights dot x features
* b: a bias
* α: the learning rate for gradient descent

## Notes
You may see a mass in the math Markdown cheatsheets, since GitHub doesn't support Math LaTex formatting. If you would like to view the LaTex formatted math cheatsheets in Markdown, plese do the followings:
* Local: In __VS code__, install __Markdown All in One__ and __Markdown Preview Enhanced__ extensions.
* Online (_GitHub_): In __Chrome__, install __MathJax Plugin for Github__ extension.