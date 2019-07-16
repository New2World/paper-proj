# Latent Relation Inference and Popularity Prediction

## Model

> TODO - illustration

## Method

Given posts, we propose a deep learning model to predict popularity of posts. Without information about users, our goals are to predict popularity of posts and learn some latent relationships among posts. We assume posts with similar topic and tags are more likely to be posted and replied by some certain users. On the other hand posts which a user prefer posting and replying to would be higher related.  
We encode a user by listing posts owned and retweeted by him/her in an one-hot vector of all posts, so that the input $X \in 2^{|\mathbb{V}|}$. Our model output a real number indicating the prediction of popularity, which is measured by number of likes and replies.  
While training we use mean squared error to measure loss and stochastic gradient descent(SGD) to optimize. The step of training is similar to expectation-maximization where we consider the structure of posts as hidden variable. Firstly we initialize $\hat{A}$ with random values and train the model to update parameters $W_{GCN}$ and $W_{FC}$ until convergence (maximization). Then we fix the parameters and use the same dataset to train the _structure matrix_ $\hat{A}$ (expectation).

## Details

- [ ] if normalization is needed
- [ ] name of $\hat{A}$
- [ ] initialization of $\hat{A}$
- [ ] range of values in $\hat{A}$