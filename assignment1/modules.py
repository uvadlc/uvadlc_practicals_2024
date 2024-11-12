################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization.
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        
        # Model parameters (weights and biases) and their gradients
        self.params = {'weight': None, 'bias': None}
        self.grads = {'weight': None, 'bias': None}

        # Kaiming Initialization
        self.params['weight'] = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        self.params['bias'] = np.zeros(out_features)
        
        # Initialize gradients as zero arrays with the same shapes as parameters
        self.grads['weight'] = np.zeros((in_features, out_features))
        self.grads['bias'] = np.zeros(out_features)

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Cache input x for use in backward pass
        self.x = x

        # Linear transformation: out = x * W + b
        out = np.dot(x, self.params['weight']) + self.params['bias']

        
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Gradient of weights and biases
        self.grads['weight'] = np.dot(self.x.T, dout)     # Shape: (in_features, out_features)
        self.grads['bias'] = np.sum(dout, axis=0)         # Shape: (out_features,)

        # Gradient with respect to the input
        dx = np.dot(dout, self.params['weight'].T)        # Shape: (batch_size, in_features)
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = None
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.x = None
        self.out = None

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        # Store x for use in the backward pass
        self.x = x
        # ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
        out = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        # Cache the output for use in the backward pass
        self.out = out
        
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        # Compute the gradient of ELU: 1 if x > 0 else alpha * exp(x)
        dx = np.where(self.x > 0, dout, dout * (self.out + self.alpha))
        
        #######################
        # END OF YOUR CODE    #
        #######################
        
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        # Clear the stored x value
        self.x = None
        self.out = None
        
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    def __init__(self):
        self.exp_x = None

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        # Subtracting the maximum value from x to avoid numerical instability
        x_max = np.max(x, axis=1, keepdims=True)
        self.x = x - x_max
        exp_x = np.exp(self.x)
        self.exp_x = exp_x
        out = exp_x / np.sum(exp_x, axis=1, keepdims=True)


        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Compute the softmax output (already stored in self.exp_x / np.sum(self.exp_x, axis=1, keepdims=True))
        softmax_out = self.exp_x / np.sum(self.exp_x, axis=1, keepdims=True)

        # Initialize the gradient dx with the same shape as dout
        dx = softmax_out * (dout - np.sum(dout * softmax_out, axis=1, keepdims=True))

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.exp_x = None
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        # Apply softmax to logits (stabilizing with max trick)
        x_exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = x_exp_shifted / np.sum(x_exp_shifted, axis=1, keepdims=True)

        # Compute the cross-entropy loss
        # Use labels y to select the correct log-probabilities
        num_samples = x.shape[0]
        log_probs = -np.log(softmax[np.arange(num_samples), y])
        out = np.sum(log_probs) / num_samples
       
        
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """


        #######################
        # PUT YOUR CODE HERE  #
        #######################


        # Compute softmax as in the forward pass
        x_exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = x_exp_shifted / np.sum(x_exp_shifted, axis=1, keepdims=True)

        # Gradient of cross-entropy loss with respect to input x
        dx = softmax
        dx[np.arange(x.shape[0]), y] -= 1  # Subtract 1 from the correct class scores
        dx /= x.shape[0]  # Average over the batch
        

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx