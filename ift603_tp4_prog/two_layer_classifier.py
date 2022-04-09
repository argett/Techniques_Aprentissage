###
#   Eliott THOMAS         —  21 164 874
#   Lilian FAVRE GARCIA   —  21 153 421
#   Tsiory Razafindramisa —  21 145 627
###

import numpy as np


class TwoLayerClassifier(object):
    def __init__(self, x_train, y_train, x_val, y_val, num_features, num_hidden_neurons, num_classes, activation='relu'):
        self.x_train = x_train  # training data
        self.y_train = y_train  # training labels
        self.x_val = x_val  # validation data
        self.y_val = y_val  # validation labels

        self.num_features = num_features  # number of features
        self.num_classes = num_classes  # number of classes

        self.net = TwoLayerNet(num_features, num_hidden_neurons, num_classes, activation)

        self.momentum_cache_v_prev = {}  # variable v of momentum

    def train(self, num_epochs=1, lr=1e-3, l2_reg=1e-4, lr_decay=1.0, momentum=0.0):
        """
        Train the model with a cross-entropy loss
        Naive implementation (with loop)

        Inputs:
        - num_epochs: the number of training epochs
        - lr: learning rate
        - l2_reg: the l2 regularization strength
        - lr_decay: learning rate decay.  Typically a value between 0 and 1
        - momentum : variable mu for momentum update

        Returns a tuple for:
        - training accuracy for each epoch
        - training loss for each epoch
        - validation accuracy for each epoch
        - validation loss for each epoch
        """
        loss_train_curve = []  # training loss for each epoch
        loss_val_curve = []  # validation loss for each epoch
        accu_train_curve = []  # training accuracy for each epoch
        accu_val_curve = []  # validation accuracy for each epoch

        self.net.reinit()  # reinitialize the weights
        self.net.l2_reg = l2_reg  # set the l2 regularization strength

        self.momentum_cache_v_prev = {id(x): np.zeros_like(x) for x in self.net.parameters}  # initialize momentum cache

        sample_idx = 0  # sample index
        num_iter = num_epochs * len(self.x_train)  # number of iterations
        for i in range(num_iter):
            # Take a sample
            x_sample = self.x_train[sample_idx]
            y_sample = self.y_train[sample_idx]  # take the label of the sample

            # Forward + Backward
            loss_train = self.net.forward_backward(x_sample, y_sample)

            # Take gradient step
            for w, dw in zip(self.net.parameters, self.net.gradients):  # update the weights
                self.momentum_update(w, dw, lr, momentum)

            # Advance in data
            sample_idx += 1
            if sample_idx >= len(self.x_train):  # End of epoch

                accu_train, loss_train = self.global_accuracy_and_cross_entropy_loss(self.x_train, self.y_train)  # training accuracy and loss
                accu_val, loss_val, = self.global_accuracy_and_cross_entropy_loss(self.x_val, self.y_val)  # validation accuracy and loss

                loss_train_curve.append(loss_train)  # store the training loss
                loss_val_curve.append(loss_val)  # store the validation loss
                accu_train_curve.append(accu_train)  # store the training accuracy
                accu_val_curve.append(accu_val)  # store the validation accuracy

                sample_idx = 0  # reset the sample index

                lr *= lr_decay  # decay the learning rate

        return loss_train_curve, loss_val_curve, accu_train_curve, accu_val_curve

    def predict(self, x):
        """
        Returns the most likely class label associated to x
        Naive implementation (with loop)
        Inputs:
        - X: A numpy array of shape (D, N) containing one or many samples.
        Returns a tuple of:
        - A numpy array of shape (N) containing one or many class label
        """
        #############################################################################
        # TODO: return the most probable class label for one sample.                #
        #############################################################################
        if len(x.shape) == 1:  # Predict on one sample
            x_inter = self.net.forward(x)  # forward pass
            class_label = np.argmax(x_inter)  # get the class label

        elif len(x.shape) == 2:  # Predict on multiple samples
            x_inter = [self.net.forward(xi) for xi in x]  # forward pass
            class_label = np.argmax(x_inter, axis=1)  # get the class label

        else:
            raise ValueError("x should be a vector or 2D matrix")

        return class_label
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

    def global_accuracy_and_cross_entropy_loss(self, x, y, l2_r=-1.0):
        """
        Compute average accuracy and cross_entropy for a series of N data
        points.
        Naive implementation (with loop)
        Inputs:
        - x: A numpy array of shape (D, N) containing several samples.
        - y: A numpy array of shape (N) labels as an integer
        - reg: (float) L2 regularization strength
        Returns a tuple of:
        - average accuracy as single float
        - average loss as single float
        """
        if l2_r > 0:
            self.net.l2_reg = l2_r  # set the l2 regularization strength
        #############################################################################
        # TODO: Compute the softmax loss & accuracy for a series of samples X,y .   #
        #############################################################################
        loss = 0
        # cross entropy
        for i in range(x.shape[0]):  # for each sample
            cross = self.net.cross_entropy_loss(self.net.forward(x[i]), y[i])  # compute the cross entropy loss
            loss += cross[0]  # add the loss

        labels = self.predict(x)  # get the predicted labels
        accu = np.mean(labels == y)  # compute the accuracy

        # Normalisation
        loss /= x.shape[0]

        return accu, loss
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

    def momentum_update(self, w, dw, lr, mu):
        """
        Compute momentum as in : http://cs231n.github.io/neural-networks-3/#sgd
        In our case, variable v is stored in self.momentum_cache_v_prev.
        The resulting w is put in parameter "w" which is passed by reference
        Returns nothing
        """

        v_prev = self.momentum_cache_v_prev[id(w)]  # get the previous velocity
        #############################################################################
        # TODO: update w with momentum                                              #
        #############################################################################

        v = mu*v_prev - lr*dw  # compute the new velocity
        # we apply the gradient momentum on v
        w += v  # update the weights

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        self.momentum_cache_v_prev[id(w)] = v


class TwoLayerNet(object):
    """
    This class encodes a network with two layers or parameters : one between the input layer
     and the hidden layer and one between the hidden layer and the output layer
    """

    def __init__(self, in_size, hidden_size, num_classes, activation='relu', l2_r=0.0):
        self.in_size = in_size
        self.num_classes = num_classes
        self.l2_reg = l2_r
        self.layer1 = DenseLayer(in_size, hidden_size, activation=activation)
        self.layer2 = DenseLayer(hidden_size, num_classes)

    def reinit(self):
        self.layer1.reinit()
        self.layer2.reinit()

    def forward(self, x):
        x1 = self.layer1.forward(x)
        x2 = self.layer2.forward(x1)
        return x2

    def backward_(self, dloss_dscores):
        dx = self.layer2.backward(dloss_dscores, self.l2_reg)
        self.layer1.backward(dx, self.l2_reg)

    def forward_backward(self, x, y):
        self.layer1.zero_grad()
        self.layer2.zero_grad()

        scores = self.forward(x)
        loss, dscores = self.cross_entropy_loss(scores, y)
        self.backward_(dscores)
        return loss

    @property
    def parameters(self):
        return [self.layer1.W, self.layer2.W]

    @property
    def gradients(self):
        return [self.layer1.dW, self.layer2.dW]

    def cross_entropy_loss(self, scores, y):
        """
        Cross-entropy loss function and gradient with respect to the score
        C.f. Eq.(4.104 to 4.109) of Bishop book.

        Inputs:
        - scores: output of the network before the softmax.  A numpy array of
        shape (C) (C is the number of classes).
        - y: training label as an integer
        Returns a tuple of:
        - loss as single float
        - gradient with respect to score; an array of same shape of scores
        """

        loss = 0
        dloss_dscores = np.zeros(np.size(scores))  # initialize the gradient with zeros

        #############################################################################
        # TODO: Compute the softmax loss and its gradient.                          #
        # Store the loss in loss and the gradient in dW.                            #
        # 1- Compute softmax => eq.(4.104) or eq.(5.25) Bishop                      #
        # 2- Compute cross-entropy loss => eq.(4.108)                               #
        # 3- Dont forget the regularization!                                        #
        # 4- Compute gradient with respect to the score => eq.(4.109) with phi_n=1  #
        #############################################################################

        # Softmax
        sfm = np.exp(scores)
        Sum = np.sum(sfm)
        sfm[:] /= Sum

        # CrossEntropy
        for i in range(len(sfm)):  # for each class
            # loss
            tnk = int(y == i)  # get the one-hot vector
            loss -= tnk*np.log(sfm[i])  # compute the cross entropy loss

            # gradient
            err = sfm[i]-tnk  # compute the error
            dloss_dscores[i] = err  # compute the gradient

        # Regularisation loss
        loss += 0.5 * self.l2_reg * np.sum(np.power(self.layer1.W, 2)) / self.in_size
        loss += 0.5 * self.l2_reg * np.sum(np.power(self.layer2.W, 2)) / self.in_size

        # Regularisation gradient
        dloss_dscores += self.l2_reg * np.sum(self.layer1.W) / self.in_size
        dloss_dscores += self.l2_reg * np.sum(self.layer2.W) / self.in_size

        return loss, dloss_dscores
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################


class DenseLayer(object):
    """
    This class encodes a layer (could be the hidden layer or the output layer)
    """
    def __init__(self, in_size, out_size, activation=None):
        self.activation = activation  # Note, 'relu', or 'sigmoid'
        self.W = None
        self.dW = None
        self.in_size = in_size  # number of input neurons
        self.out_size = out_size  # number of output neurons
        self.reinit()

        self.last_x = None
        self.last_activ = None

    def reinit(self):
        self.W = np.random.randn(self.in_size + 1, self.out_size) * np.sqrt(0.1 / (self.in_size + self.out_size))
        self.dW = np.zeros_like(self.W)

    def zero_grad(self):
        self.dW.fill(0.0)

    def forward(self, x):
        """
        Computer forward pass for 1 layer :

                f=h(dot(W,x))

        where h is the layer's activation function.

        Inputs:
        - x: A numpy array of shape (D)
        Returns a tuple of:
        - f: a floating point value
        """
        x = augment(x)  # add the bias term
        #############################################################################
        # TODO: Compute forward pass.  Do not forget to add 1 to x in case of bias  #
        # C.f. function augment(x)                                                  #
        #############################################################################
        dot = x@self.W
        if self.activation == 'sigmoid':
            dot = sigmoid(dot)
        elif self.activation == 'relu':
            dot = reLU(dot)
        else:  # no activation function equals linear
            dot = dot

        f = dot
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        self.last_x = x
        self.last_activ = f

        return f

    def backward(self, dnext_dout, l2_reg):
        if self.activation == 'sigmoid':
            dout_dsigmoid = self.last_activ * (1.0 - self.last_activ)
            dnext_dsigmoid = dnext_dout * dout_dsigmoid  # type:np.ndarray
            dnext_dW = self.last_x[:, np.newaxis] * dnext_dsigmoid[np.newaxis, :]
            dnext_dX = dnext_dsigmoid.dot(self.W.T)
        elif self.activation == 'relu':
            dout_drelu = self.last_activ != 0.0
            dnext_drelu = dnext_dout * dout_drelu  # type:np.ndarray
            dnext_dW = self.last_x[:, np.newaxis] * dnext_drelu[np.newaxis, :]
            dnext_dX = dnext_drelu.dot(self.W.T)
        else:
            dnext_dW = self.last_x[:, np.newaxis].dot(dnext_dout[np.newaxis, :])
            dnext_dX = dnext_dout.dot(self.W.T)
        dnext_dX = dnext_dX[:-1]  # discard the gradient wrt the 1.0 of homogeneous coord

        self.dW += dnext_dW
        self.dW += l2_reg * self.W  # add regul. gradient
        return dnext_dX


def augment(x):
    if len(x.shape) == 1:
        return np.concatenate([x, [1.0]])
    else:
        return np.concatenate([x, np.ones((len(x), 1))], axis=1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def reLU(x):
    if len(x.shape) == 1:  # if x is a vector
        return np.array([xi if xi > 0 else 0 for xi in x])
    else:  # if x is a matrix
        for i in range(x.shape[0]):  # apply ReLU on each row
            for j in range(x.shape[1]):  # apply ReLU on each column
                if x[i, j] < 0:
                    x[i, j] = 0
        return x
