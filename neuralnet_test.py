#!/usr/bin/env python

import random
import numpy as np
import neuralnet as nn
import matplotlib.pyplot as plt

FILEPATH_THETAS = "thetas_sinus_1_20_1.pkl"
PI_CYCLES = 2


# Generate a random number where:  a <= rand < b
def rand(a, b):
    return (b - a) * random.random() + a

#
# Testing methods using different learning data.
# Please contribute with more tests.
#


# Test NeuralNet using sinus data.
# 'num_training' - number of training samples
# 'num_testing' - number of testing samples
# 'learn' - a boolean showing whether to learn new weights. If 'learn' is False, weights saved in pickle file are used
# to evaluate the NeuralNet
def test_sinus_data(num_training, num_testing, learn=True):
    # change parameters here
    shape = [1, 20, 1]
    _lambda = 0
    net = nn.NeuralNet(shape=shape, _lambda=_lambda)

    # Use weights from file for faster convergence
    # net.init_weights = net.load_thetas(FILEPATH_THETAS)

    # Learn thetas and evaluate
    if learn:
        # Generate (unnormalised) data
        pi_cycles = 2
        Xlun = [rand(-3.14 * pi_cycles, 3.14 * pi_cycles) for i in range(num_training)]
        Xlun = sorted(Xlun)
        Yl = np.sin(Xlun)
        Xlun = np.resize(Xlun, (num_training, 1))
        Yl = np.resize(Yl, (num_training, 1))

        # Normalise data
        Xln = (Xlun + 3.14 * pi_cycles) / (2 * 3.14 * pi_cycles)  # normalize in [0,1]
        Xln2 = (Xlun - np.mean(Xlun)) / (max(Xlun) - min(Xlun))  # normalize, mean zero
        Yln = (Yl + 1) / 2

        # Learn using scipy. Save learned weights with pickle
        train_result = net.train_scipy(X=Xln, Y=Yln)
        net.filepath_picklethetas = FILEPATH_THETAS
        net.pickle_thetas()
    else:
        # Use weights from file
        net.weights = net.load_thetas(FILEPATH_THETAS)
        train_result = None

    # Generate data for testing
    Xeun = [rand(-3.14 * PI_CYCLES, 3.14 * PI_CYCLES) for _ in range(num_testing)]
    Xeun = sorted(Xeun)
    Ye = np.sin(Xeun)
    Xeun = np.resize(Xeun, (num_testing, 1))
    Ye = np.resize(Ye, (num_testing, 1))

    # Normalise evaluation data
    Xen = (Xeun + 3.14 * PI_CYCLES) / (2 * 3.14 * PI_CYCLES)  # normalise in [0,1]
    # Xen2 = (Xeun - np.mean(Xeun)) / (max(Xeun) - min(Xeun))  # mean zero normalisation
    Yen = (Ye + 1) / 2

    # Test
    Jtrain, predictions = net.test(Xen, Yen)

    # Un-normalize data
    preds_un = (predictions * 2) - 1

    # Print details
    print('Test details:')
    print('-------------------')
    print("NN shape,:", net.shape)
    print("Training samples:", num_training)
    print("Testing samples:", num_testing)
    print("Lambda:", net._lambda)
    print("init_weights interval: [%.2f, %.2f]" % (np.min(nn.wrap_thetas(net.init_weights)),
                                                   np.max(nn.wrap_thetas(net.init_weights))))
    print("Method: scipy.optimize.minimize (BFGS)")
    print("Data normalisation: input: normalised [0,1] / output: [0,1]")
    print("Train error:", train_result.fun if train_result else None)
    print("Test error:", Jtrain)
    print('-----------------------------------------------------------')

    plt.plot(Xeun, Ye, "-", label="sin(x)")
    plt.plot(Xeun, preds_un, "x", label="estimation")
    plt.xlabel('x')
    plt.legend()
    plt.savefig("sinus_test.png")
    plt.show()


def main():
    test_sinus_data(num_training=5000, num_testing=1000, learn=False)


if __name__ == '__main__':
    main()
