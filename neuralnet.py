import os
import pickle
import numpy as np
from scipy.optimize import minimize


class NeuralNet:
    """ Back propagation neural network"""

    #
    # Class members
    #
    shape = None
    layer_count = 0
    weights = []
    init_weights = []

    #
    # Class methods
    #
    def __init__(self, shape, init_weights=None, _lambda=None):
        """Initialisation"""

        # Cost function J
        self.J = None

        # Number of layers (not counting the first)
        self.layer_count = len(shape) - 1

        # Shape of the NeuralNet (input, hidden and output layers)
        self.shape = shape

        # Regularisation parameter _lambda
        self._lambda = _lambda if _lambda is not None else 0

        # Input / output data from last run
        # todo: I don't need this
        self._layer_input = []
        self._layer_output = []

        # Initialise weights
        if init_weights is None:
            self.init_weights = initialise_weights(shape)
        else:
            self.init_weights = init_weights
        self.weights = self.init_weights

        # Filepaths used to save data
        self.filepath_pickleself = "my_nn.pkl"
        self.filepath_itertext = "iterations.txt"
        self.filepath_picklethetas = "thetas.pkl"

        # Open a text file to save the cost function of each iteration
        file = open(self.filepath_itertext, 'w')
        file.close()

    # Forward propagation
    def forward_prop(self, thetas, x):
        activations = [0] * (self.layer_count + 1)
        activations_bias = [0] * self.layer_count
        activations[0] = x
        for i in range(self.layer_count):
            activations_bias[i] = np.concatenate((np.ones((activations[i].shape[0], 1)), activations[i]), axis=1)
            activations[i + 1] = sigmoid(np.dot(activations_bias[i], thetas[i].T))
        hypothesis = activations[-1]
        for i in range(len(hypothesis)):
            if hypothesis[i] == 1:
                hypothesis[i] = 1 - 1e-10
        return hypothesis, activations_bias

    # Calculate the cost function
    def cost(self, thetas, y, hypothesis):
        num_samples = len(y)
        J = y * np.log(hypothesis) + (1 - y) * np.log(1 - hypothesis)
        J = (-1) * np.sum(J) / num_samples
        reg = 0
        for weight in thetas:
            reg += np.sum(weight[:, 1:] ** 2)
        reg *= self._lambda / (2 * num_samples)
        J += reg
        return J

    # Back-propagation
    def back_prop(self, thetas, y, hypothesis, _activations):

        num_samples = len(y)
        # Initialize values - cumulative thetas
        Thetas = [0] * self.layer_count
        for i, t in enumerate(self.weights):
            Thetas[i] = np.zeros_like(thetas[i])

        # Back propagation
        errors = [0] * self.layer_count
        errors[-1] = hypothesis - y
        if self.layer_count > 1:
            for i in range(self.layer_count):
                k = self.layer_count - i - 1
                errors[k - 1] = np.dot(errors[k], thetas[k][:, 1:]) * sigmoid(_activations[k][:, 1:], deriv=True)
                for j in range(num_samples):
                    err_line = np.reshape(errors[k][j, :], (len(errors[k][j, :]), 1))
                    act_line = np.reshape(_activations[k][j, :], (len(_activations[k][j, :]), 1))
                    Thetas[k] = Thetas[k] + np.dot(err_line, act_line.T)
        else:
            for j in range(num_samples):
                err_line = np.reshape(errors[0][j, :], (len(errors[0][j, :]), 1))
                act_line = np.reshape(_activations[0][j, :], (len(_activations[0][j, :]), 1))
                Thetas[0] = Thetas[0] + np.dot(err_line, act_line.T)

        # Calculate derivatives
        derivs = [0] * self.layer_count
        for i in range(self.layer_count):
            derivs[i] = (Thetas[i] + self._lambda * thetas[i]) / num_samples
            derivs[i][:, 0] = Thetas[i][:, 0] / num_samples

        derivs = np.array(wrap_thetas(derivs))
        return derivs

    # An iteration of forward-back propagation. Returns the cost function J and the weights partial derivatives
    def iterate(self, thetas_list, x, y):

        thetas = unwrap_thetas(thetas_list, self.shape)

        # Forward propagation
        hypothesis, _activations = self.forward_prop(thetas, x)

        # Cost function
        J = self.cost(thetas, y, hypothesis)

        # Back propagation
        derivs = self.back_prop(thetas, y, hypothesis, _activations)

        return J, derivs

    # Uses iterate method to return the cost function J. Used in train_scipy method to mimimise J.
    def get_J(self, thetas_list, x, y):
        J = self.iterate(thetas_list, x=x, y=y)[0]
        print("J =", J)
        self.J = J
        self.cost_to_text()
        return J

    # Uses iterate method to return the weights derivatives. Used in train_scipy method to mimimise J.
    def get_derivs(self, thetas_list, x, y):
        return self.iterate(thetas_list, x=x, y=y)[1]

    # Method that uses scipy.optimization to minimize J
    # Works well with BFGS method, as weights_interval is larger it takes more time to converge, but succeeds.
    def train_scipy(self, X, Y):
        print("Learning (BFGS)..")
        print("---------------")
        init_weights_list = wrap_thetas(self.init_weights)
        res = minimize(self.get_J, init_weights_list, method='BFGS', jac=self.get_derivs, options={'disp': True},
                       args=(X, Y))
        self.weights = unwrap_thetas(res.x, self.shape)
        return res

    # Save the cost function of each iteration to text file
    def cost_to_text(self):
        with open(self.filepath_itertext, 'a') as file:
            file.write("J = %.5f\n" % self.J)
            file.close()

    # Save the NeuralNet to a file using pickle
    def pickle_self(self):
        with open(self.filepath_pickleself, 'wb') as file:
            pickle.dump(self, file)

    # Save the (learned) weights to file using pickle
    def pickle_thetas(self):
        with open(self.filepath_picklethetas, 'wb') as file:
            pickle.dump(self.weights, file)

    # Load weights from file using pickle.
    # Don't forget to check if the size of the weights is same as self.shape
    def load_thetas(self, filepath):
        with open(filepath, 'rb') as file:
            return pickle.load(file)


#
# Other useful methods
#

# Sigmoid function, and its derivative
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# Initialisation of weights with shape 'nn_shape' in interval: [-upper_interval, upper_interval]
def initialise_weights(nn_shape, upper_interval=None):
    np.random.seed(1)
    weights = []
    for l1, l2 in zip(nn_shape[:-1], nn_shape[1:]):
        if upper_interval is not None:
            weights.append(np.random.random(size=(l2, l1 + 1)) * 2 * upper_interval - upper_interval)  # mean should be around 0
        else:
            weights.append(np.random.random(size=(l2, l1 + 1)))  # interval [0,1)
    return weights


# Concatenate list of arrays into a single list of values
def wrap_thetas(thetas):
    thetas_list = []
    for theta in thetas:
        theta = theta.T  # so wrapping works same as Matlab
        r, c = theta.shape
        thetas_list.extend(np.reshape(theta, (r * c,)).tolist())
    return np.array(thetas_list)


# Get a list of arrays (according to network architecture) from a single list of values
def unwrap_thetas(thetas_list, shape):
    _last_index = 0
    thetas = []
    for i in range(len(shape) - 1):
        _in = shape[i] + 1
        _out = shape[i + 1]
        _index = (_in * _out) + _last_index
        theta = np.reshape(np.asarray(thetas_list[_last_index:_index]), (_in, _out))
        thetas.append(theta.T)  # Transpose so it works same as in Andrew Ng class (in Matlab)
        _last_index = _index
    return thetas

#
# If run as a script
#
if __name__ == "__main__":

    #
    # A simple test
    #
    nn = NeuralNet((2, 4, 1))
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    Y = np.array([[0.5, 1, 1, 0.5]]).T
    nn.train_scipy(X, Y)
    print("prediction for [0,0]:", nn.forward_prop(nn.weights, np.array([[0, 0]]))[0])
    print("prediction for [0,1]:", nn.forward_prop(nn.weights, np.array([[0, 1]]))[0])
    print("prediction for [1,0]:", nn.forward_prop(nn.weights, np.array([[1, 0]]))[0])
    print("prediction for [1,1]:", nn.forward_prop(nn.weights, np.array([[1, 1]]))[0])