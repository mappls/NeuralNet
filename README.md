# NeuralNet
Neural network class created from scratch

This is my custom-made Python class of a Nerual Network using the backpropagation learning algorithm. All coding is created from scratch, according to Andrew Ng's Coursera class "Machine Learning".


## Brief description of the class attributes and methods

A NeuralNet object created using this class has the following attributes (only the most important are listed):
- 'shape' - the shape of the network, eg. 2-10-1 is a 3-layered network with 2 inputs, 1 output and 10 hidden nodes,
- 'name' - name of the network, useful when comparing multiple objects,
- 'layer_count' - the number of layers,
- 'init_weights' - initial weights of all connections in the network, before training it,
- 'weights' - the learned weights of connections in the network,
- 'lambda_' - a regularization parameter,
- 'J' - the cost function that needs to be minimised
- 'Jtest' - the cost function calculated on the test set

And the following (most important only) methods:
- 'train_scipy(Xtrain, Ytrain)' - train the network on 'Xtrain' data, with 'Ytrain' targets. Calculates the value of the cost function 'J',
- 'test(Xtest, Ytest)' - evaluate the algorithm on the test set 'Xtest / Ytest'. Calculates the cost function 'Jtest',
- 'json_self(filepath)' - saves the whole object to a JSON file with path 'filepath',
- 'load_json(filepath, nn_dict) - load a NeuralNet object from file path 'filepath' ('nn_dict' = None in this case), or from another dictionary object ('nn_dict' = <dictionary object>  in this case),


## Description of the files in the repository

- neuralnet.py - the main class of the project,
- neuralnet_test.py - a script for testing the NeuralNet class on the sinus function,
- thetas_sinus_1_20_1.pkl - a Pickle file containing the weights learned by the network (shape 1-20-1) for a sinus function test,
- sinus_test.png - a visual representation of the network fit to actual sinus data.
