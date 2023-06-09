# Neural Network

This program trains and evaluates a neural network for multi-class classification. 
This program allows the user to specify the structure of a multi-layer feed-forward 
neural network and then will train that network using backpropagation in conjunction 
with mini-batch gradient descent.

## Usage

A user can run this program with the command:

```bash
java NeuaralNetDriver -f [TRAINING_FILE]
```

in the directory containing this program's class files. This program takes 
one required and several optional command line arguments:

-f [TRAINING_FILE]: Reads training data from the file named [TRAINING_FILE] (specified as a 
					String).

-h <NH> <S1> <S2> .. <INTEGER>: Specifies the number of hidden layers <NH> followed by <NH> additional 
								integers corresponding to the sizes of these hidden layers. The following 
								integer(s) <S1>, <S2>, etc. represent the sizes of hidden layer 1, hidden 
								layer 2, and so on. Default is 0.

-a  <DOUBLE>: Specifies the learning rate alpha in mini-batch gradient descent; default is 0.01.

-l <INTEGER>: Specifies the limit for training group size; default is 100.

-e <INTEGER>: Specifies the epoch limit in mini-batch gradient descent; default is 1000.

-m <INTEGER>: Specifies the batch size in mini-batch gradient descent; default is 1 for stochastic 
			  gradient descent (using -m 0 should be interpreted as full batch gradient descent).

-l  <DOUBLE>: Specifies the regularization hyperparameter lambda; default is 0.0 (no regularization).

-r          : If specified, this flag (which has no arguments) enables randomization of data for the
			  train/validation split and for batch construction at the start of each epoch; if this 
			  flag is not specified, then data will not be randomized.

-w  <DOUBLE>: Specifies the value epsillon for weight initialization; default is 0.1.

-v <INTEGER>: Specifies a verbosity level, indicating how much output the program should produce; 
			  default is 1.

## Bugs and Unimplemented Functionality

None

## Author
John Cornwell

Date: 11/12/2022

CS 557: Machine Learning
