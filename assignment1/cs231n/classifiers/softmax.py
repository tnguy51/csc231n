import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dW = np.zeros_like(W)


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  for i in range(num_train):
    scores = X[i].dot(W)

    # Calculate the normalization factor
    prob_sum = 0
    for j in range(num_classes):
      prob_sum += np.exp(scores[j])

    # Caculate the loss and  gradient
    loss += -scores[y[i]] + np.log(prob_sum)
    for j in range(num_classes):
      dW[:, j] += X[i]*np.exp(scores[j])/prob_sum
    dW[:, y[i]] -= X[i]

  loss /= num_train
  dW /= num_train

  # regularization
  loss  += reg*np.sum(W**2)
  dW += 2*reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_train = X.shape[0]
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  scores = X.dot(W)
  prob_sum = np.sum(np.exp(scores), axis=1)
  loss = np.sum(-scores[np.arange(num_train), y] + np.log(prob_sum))

  mask = np.exp(scores)/prob_sum[:, None]
  mask[np.arange(num_train), y] -= 1
  dW = X.T.dot(mask)

  loss /= num_train
  dW /= num_train

  # regularization
  loss  += reg*np.sum(W**2)
  dW += 2*reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

