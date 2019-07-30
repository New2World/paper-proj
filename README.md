# Latent Relation Inference and Popularity Prediction

_Assumption_: posts with similar topic and tags are more likely to be posted and replied by some certain users. On the other hand posts which a user prefer posting and replying to would be highly related.

## Model

![gcn_model](res/gcn_model.png)

Given some information about posts from online social networks like Reddit, we propose a deep learning model to predict popularity of posts. Our model architecture consists of three components: a post encoder ![rf1], a context encoder ![rf2] and a regressor ![rf3].

The **post encoder** is a 3-layer graph convolutional network (GCN), which take as input of a 2-D one-hot matrix where each row denotes a post and each row is a user so that the input is ![rf4]. This input matrix shows how many and which users are involved in a certain post. Recall the definition of GCN, according to our assumption we initialize the adjacent matrix as inner product of our input which intuitively means the correlationship between posts. The 3-layer GCN will output a embedding representation ![rf5] for each post, where ![rf6] is the dimension of embedding.

![rf7]

The **context encoder** use post context information to build the correlationship between posts. To avoid the variance of learning word embedding from scratch and the bias of introducing existing word2vec, we train our context encoder in character level. Characters are transformed into one-hot vectors so we can get a sequence of one-hot vectors ![rf8] for each post. Using the model proposed by _X Zhang et, al. 2015_, a 1-D convolutional network is applied to the input to extract text features and output post embedding ![rf9].

By concatenating outputs from above two encoders, the **regressor** learns with information from both user aspect and context. We build 3 fully connected layers to learn the popularity of each post given the cancatenated embedding representations, so the output is vector ![rf10] and each dimension is a real number indicating the prediction of popularity.

## Training

Our goals are to 1) predict popularity of each post; 2) learn the latent relationship between posts, so we have two group of parameters should be learned. First are those weights between GCN, CNN and fully connected layers, and second is the adjacent matrix. However, the gradient directions of first group of parameters largely depend on the state of adjacent matrix, which means we can not update them simultaneously. We propose a training method based on expectation-maximization (EM) algorithm and update these parameters in two separated steps.

During maximization phrase, we first generate the input of post encoder as one-hot matrix ![rf11] and initialize the adjacent matrix ![rf16] in GCN with the inner product of ![rf12]. Meanwhile we map characters in each post to one-hot vectors and input the sequence of one-hot vectors into our context encoder. After forward propagation we get two distinguished embedding. Concatenate embeddings post by post (row by row) we get a new matrix with size ![rf13], and this will be the input of regressor. We measure our loss by calculating mean squared error (MSE) between the prediction and ground truth among all posts. By backpropagation, gradients are propagated back to the very beginning and all parameters except the adjacent matrix are updated to minimize the loss. This training process will be performed until the model converge.

In expectation phrase, all inputs and outputs are the same with those in maximization phrase, however, the only difference is the parameters that need to be updated. Our maximization phrase is based on the correlation assumption encoded in the adjacent matrix, so in expectation step it is the adjacent matrix that should be updated, while other parameters are fixed.

## Experiment

### Convergency

First we should make sure the EM process converge after limited iterations. We only use the post encoder to generate embeddings ![rf14], input a one-hot matrix that each row represents a post by activating all users involved and each column shows all posts a user participate in and output post embeddings. The adjacent matrix in GCN is initialized by the correlation matrix of our input. Loss of this model is the mean squared error between the learned adjacent matrix and the correlation matrix represented by the embedding. We use this loss in both expectation and maximization phrases.

![rf15]

### Popularity prediction with GCN only

To evaluate the performance of GCN in capturing relation of posts and predicting popularity, we apply fully connected layers right after our post encoder. With the same input, the fully connected network will output one real number which is the prediction of a certain post. As our experiment suggests, the learning rates for expectation and maximization phrases are different. Empirically, the update of adjacent matrix in each step is smaller than that of other parameters so we set the learning rate in expectation phrase to 0.0001 while for maximization it is 0.001.

## Results

### Convergency

The loss keeps decreasing from 0.032 to 1e-18 in both expectation and maximization phrases. We get the correlation matrix of posts from the inner product of output embeddings. As shown in Figure 1, the light points means higher difference while the dark parts means lower difference.

### Popularity prediction with GCN only

Popularity prediction is a regression task so the evaluation metric we use is mean absolute error (MAE). We perform 10 iteration of EM process, and in each iteration expectation runs 50 epochs and maximization runs 10 epochs. The final MAE on Reddit testing set is 5.71.

![popularity prediction 100](res/popularity_prediction_100.png)

![initial adjacent matrix](res/initial_adjacent_matrix.png)

![learned adjacent matrix](res/learned_adjacent_matrix.png)

[rf1]: http://chart.apis.google.com/chart?cht=tx&chl=E_{POST}
[rf2]: http://chart.apis.google.com/chart?cht=tx&chl=E_{CONTEXT}
[rf3]: http://chart.apis.google.com/chart?cht=tx&chl=R
[rf4]: http://chart.apis.google.com/chart?cht=tx&chl=X_{POST}\in\{0,1\}^{|M|\times|N|}
[rf5]: http://chart.apis.google.com/chart?cht=tx&chl=H_{POST}\in\mathbb{R}^{d_{POST}}
[rf6]: http://chart.apis.google.com/chart?cht=tx&chl=d_{POST}
[rf7]: http://chart.apis.google.com/chart?cht=tx&chl=H^{(l+1)}=\sigma[\hat{D}^{-\frac12}\hat{A}\hat{D}^{-\frac12}H^{(l)}W^{(l)}]
[rf8]: http://chart.apis.google.com/chart?cht=tx&chl=X_{CONTEXT}
[rf9]: http://chart.apis.google.com/chart?cht=tx&chl=H_{CONTEXT}\in\mathbb{R}^{d_{CONTEXT}}
[rf10]: http://chart.apis.google.com/chart?cht=tx&chl=\hat{y}\in\mathbb{R}^{|M|}
[rf11]: http://chart.apis.google.com/chart?cht=tx&chl=X_{POST}\in\mathbb{R}^{|M|\times|N|}
[rf12]: http://chart.apis.google.com/chart?cht=tx&chl=X_{POST}
[rf13]: http://chart.apis.google.com/chart?cht=tx&chl=|M|\times(d_{POST}+d_{CONTEXT})
[rf14]: http://chart.apis.google.com/chart?cht=tx&chl=H
[rf15]: http://chart.apis.google.com/chart?cht=tx&chl=loss=MSE(HH^T-A)
[rf16]: http://chart.apis.google.com/chart?cht=tx&chl=\hat{A}