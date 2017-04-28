import os
import json
import numpy as np
from scipy.optimize import minimize
from collections import OrderedDict


class NeuralNet:
    """ Back propagation neural network"""

    #
    # Class members
    #
    J = None
    Jtest = None
    name = None
    _lambda = 0
    weights = []
    shape = None
    layer_count = 0
    init_weights = []
    filepath_itertext = None
    filepath_jsonself = None


    #
    # Class methods
    #
    def __init__(self, shape, name=None, init_weights=None, _lambda=None):
        """Initialisation"""

        # Network name
        self.name = name if name is not None else "default"

        # Cost function J, test cost function Jtest
        self.J = None
        self.Jtest = None

        # Number of layers (not counting the first)
        self.layer_count = len(shape) - 1

        # Shape of the NeuralNet (input, hidden and output layers)
        self.shape = shape

        # Regularisation parameter _lambda
        self._lambda = _lambda if _lambda is not None else 0

        # Initialise weights
        if init_weights is None:
            self.init_weights = initialise_weights(shape)
        else:
            self.init_weights = init_weights
        self.weights = self.init_weights

        # Filepaths used to save data
        self.filepath_jsonself = "nn_" + self.name + ".json"
        self.filepath_itertext = "iters_" + self.name + ".txt"

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

    # An iteration of forward-back propagation. Returns the cost function J and the weights' partial derivatives
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

    # Test the algorithm using the learned weights
    # Xt, Yt - the input and output (correct) data
    # Returns the test error Jtest, and the predicted values (hypothesis)
    def test(self, Xt, Yt):
        hypothesis, _ = self.forward_prop(self.weights, Xt)
        self.Jtest = self.cost(self.weights, Yt, hypothesis)
        return self.Jtest, hypothesis

    # Save the cost function of each iteration to text file
    def cost_to_text(self):
        with open(self.filepath_itertext, 'a') as file:
            file.write("J = %.5f\n" % self.J)
            file.close()

    # Load weights from file using JSON.
    # Don't forget to check if the size of the weights is same as self.shape
    def load_thetas(self, filepath):
        n = NeuralNet([])
        n.json_load(filepath)
        return n.weights

    # Save object to JSON
    def json_self(self, filepath=None):
        nn_dict = {}
        nn_dict['name'] = self.name
        nn_dict['shape'] = self.shape
        nn_dict['lambda'] = self._lambda
        nn_dict['J'] = self.J
        nn_dict['Jtest'] = self.Jtest
        nn_dict['layer_count'] = self.layer_count
        nn_dict['filepath_itertext'] = self.filepath_itertext
        nn_dict['filepath_jsonself'] = self.filepath_jsonself

        if len(self.weights) > 0:
            nn_dict['weights'] = [w.tolist() for w in self.weights]
        else:
            nn_dict['weights'] = None

        if len(self.init_weights) > 0:
            nn_dict['init_weights'] = [w.tolist() for w in self.init_weights]
        else:
            nn_dict['init_weights'] = None

        if filepath is not None:
            self.filepath_jsonself = filepath
            with open(filepath, 'w') as file:
                json.dump(nn_dict, file)

        return json.dumps(nn_dict)

    # Load a NeuralNet from JSON file
    def json_load(self, filepath):
        if filepath is not None:
            with open(filepath, 'r') as file:
                nn_dict = json.load(file)
            self.name = nn_dict['name']
            self.J = nn_dict['J']
            self.Jtest = nn_dict['Jtest']
            self._lambda = nn_dict['lambda']
            self.shape = nn_dict['shape']
            self.layer_count = nn_dict['layer_count']
            self.filepath_itertext = nn_dict['filepath_itertext']
            self.filepath_jsonself = nn_dict['filepath_jsonself']

            if len(nn_dict['weights']) > 0:
                self.weights = []
                for w in nn_dict['weights']:
                    self.weights.append(np.asarray(w))
            else:
                self.weights = np.asarray(nn_dict['weights'])

            if len(nn_dict['init_weights']) > 0:
                self.init_weights = []
                for w in nn_dict['init_weights']:
                    self.init_weights.append(np.asarray(w))
            else:
                self.init_weights = np.asarray(nn_dict['init_weights'])
            return self
        return None


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
    np.random.seed(1)  # todo: change this
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

    # Create a NeuralNet object and generate some data (XOR function)
    nn = NeuralNet((2, 4, 1))
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    Y = np.array([[0, 1, 1, 0]]).T

    # Train the network and print result
    res = nn.train_scipy(X, Y)
    if res.success:
        print("Train error:", res.fun)

    # Test using a single sample
    Jtest, _ = nn.test(np.array([X[0]]), np.array([Y[0]]))
    print("Test error:", Jtest)
    print("prediction for [0,0]:", nn.forward_prop(nn.weights, np.array([[0, 0]]))[0])
    print("prediction for [0,1]:", nn.forward_prop(nn.weights, np.array([[0, 1]]))[0])
    print("prediction for [1,0]:", nn.forward_prop(nn.weights, np.array([[1, 0]]))[0])
    print("prediction for [1,1]:", nn.forward_prop(nn.weights, np.array([[1, 1]]))[0])