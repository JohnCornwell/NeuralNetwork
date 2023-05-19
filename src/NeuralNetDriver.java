// John Cornwell

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;

/*
 * @Author John Cornwell
 * 
 * @Version 1.0
 * 
 * This class is responsible for training and evaluating a neural 
 * network for multi-class classification. This driver will allow the 
 * user to specify the structure of a multi-layer feed-forward neural 
 * network and then will train that network using back propagation in 
 * conjunction with mini-batch gradient descent.
 */
public class NeuralNetDriver {

	/*
	 * This class represents a single node in a feed forward neural network.
	 */
	private class Node {

		// the weights on the edges connecting parents to this node
		private double[] weights;
		// stores delta j * ai sums for weight with the same index
		private double[] derivatves;
		// the value of the inputs multiplied by their respective weights
		private double input;
		// the value that this node produces
		private double output;
		// delta j is the partial derivative of the cost function with respect
		// to this node's input
		private double delta;

		/*
		 * Sets the starting weights. For a node not in the input layer, the
		 * first weight will correspond to the bias node. All subsequent weights
		 * correspond to the parent nodes from the previous layer in order.
		 */
		public void initWeights(double[] weights) {
			this.weights = weights;
			this.derivatves = new double[this.weights.length];
		}

		/*
		 * This method will produce an output by applying this nodes activation
		 * function on the product of the weights and the previous layer's
		 * output (and the bias node if it is not in the previous layer).
		 */
		public void input(Node[] parents, Node bias) {
			this.input = 0.0;
			if (bias != null) {
				// common case where parent layer is a hidden layer
				this.input += bias.output * this.weights[0];
				for (int i = 1; i < this.weights.length; i++) {
					// multiply parent's weight by its output
					this.input += this.weights[i] * parents[i - 1].output;
				}
			} else {
				// parent layer is the input layer
				for (int i = 0; i < this.weights.length; i++) {
					// multiply parent's weight by its output
					this.input += this.weights[i] * parents[i].output;
				}
			}
			this.output = activation(this.input);
		}

		/*
		 * This variant of input is used on input nodes. The output of the node
		 * after this operation will be the input value given.
		 */
		public void input(double in) {
			this.output = in;
		}

		/*
		 * The activation function for this node.
		 */
		public double activation(double input) {
			return 1.0 / (1.0 + Math.pow(Math.E, -1.0 * input));
		}

		public void zeroDerivatives() {
			for (int i = 0; i < this.derivatves.length; i++) {
				this.derivatves[i] = 0.0;
			}
		}
	}

	private BufferedReader trainingFile = null;
	private String trainingFileName;
	private ArrayList<double[]> datapoints; // all data from file
	private ArrayList<double[]> trainingSet; // regularized for training
	private ArrayList<double[]> validationSet; // regularized for validation
	// specifies the number of nodes per hidden layer
	private int[] layers = new int[0];
	private double learningRate = 0.01;
	private int epochLimit = 1000;
	private int batchSize = 1;
	private double regularization = 0.0;
	private boolean randomization = false;
	private double weights = 0.1;
	private int verbosity = 1;
	// the number of possible classifications for a data point
	private int classes;
	private ArrayList<Node[]> network;
	private Node[] outputLayer;
	// allowable error at any output node
	private static final double PRECISION = 0.01;

	/*
	 * This method executes evaluation of a point on the network using forward
	 * propagation. backProp is true when this method is called as part of a
	 * backWards propagation run. Only then will the verbosity 3 print
	 * statements be possible.
	 */
	public void forwardPropagate(double[] point, boolean backProp) {
		// forward propagation
		Node[] input = this.network.get(0);
		for (int i = 0; i < input.length; i++) {
			// input includes bias
			input[i].input(point[i]);
		}
		if (this.verbosity > 3 && backProp) {
			System.out.print("      Layer 1 (input) :      a_j:");
			for (int i = 0; i < input.length; i++) {
				System.out.print(String.format("%7.3f", input[i].output));
			}
			System.out.println();
		}
		Node[] parent = input;
		Node[] current;
		for (int i = 1; i < this.network.size(); i++) {
			current = this.network.get(i);
			Node biasNode = (i == 1) ? null : input[0];
			for (Node k : current) {
				k.input(parent, biasNode);
			}
			if (this.verbosity > 3 && backProp) {
				if (i != this.network.size() - 1) {
					System.out.print(
							"      Layer " + (i + 1) + " (hidden):     in_j:");
				} else {
					System.out.print(
							"      Layer " + (i + 1) + " (output):     in_j:");
				}
				for (int j = 0; j < current.length; j++) {
					System.out.print(String.format("%7.3f", current[j].input));
				}
				System.out.print("\n                             a_j:");
				for (int j = 0; j < current.length; j++) {
					System.out.print(String.format("%7.3f", current[j].output));
				}
				System.out.println();
			}
			parent = current;
		}
		if (this.verbosity > 3 && backProp) {
			System.out.print("              example's actual y:");
			for (int k = 0; k < this.classes; k++) {
				double val = (k == (int) point[point.length - 1]) ? 1.0 : 0.0;
				System.out.print(String.format("%7.3f", val));
			}
			System.out.println();
		}
	}

	/*
	 * This function gets the average loss of the network on the given data set.
	 */
	public double getLoss(ArrayList<double[]> data) {
		double lossSum = 0.0;
		double prediction;
		Node[] outputLayer = this.outputLayer;
		double loss; // loss for a point
		double[] point;
		// sum of loss
		for (int i = 0; i < data.size(); i++) {
			// for all data points
			point = data.get(i);
			// evaluate the point
			this.forwardPropagate(point, false);
			loss = 0.0;
			// squared loss (yk - ak)^2 for all classes
			for (int k = 0; k < this.classes; k++) {
				prediction = outputLayer[k].output;
				// if the index is yk, we square 1 - prediction,
				// else we square the prediction
				loss += (k == (int) point[point.length - 1])
						? Math.pow(1.0 - prediction, 2)
						: Math.pow(prediction, 2);
			}
			lossSum += loss;
		}
		return (lossSum / data.size());
	}

	/*
	 * This method will compute the current cost of a set of weights on a given
	 * data set.
	 */
	public double getCost(ArrayList<double[]> data) {
		double weightCost = 0.0;
		double averageLoss = getLoss(data);
		// get weights from all layers except the input layer
		if (this.regularization != 0.0) {
			for (int i = 1; i < this.network.size(); i++) {
				Node[] layer = this.network.get(i);
				for (int j = 0; j < layer.length; j++) {
					// for every node
					double[] nodeWeights = layer[j].weights;
					// sum of the square of all weights
					for (double w : nodeWeights) {
						weightCost += Math.pow(w, 2);
					}
				}
			}
		}
		return averageLoss + (this.regularization * weightCost);
	}

	/*
	 * This method evaluates the accuracy of the network on the training and
	 * validation sets.
	 */
	public void evaluate() {
		System.out.println("* Evaluating accuracy");
		double trainingError = getAccuracy(this.trainingSet);
		double validationError = getAccuracy(this.validationSet);
		System.out.println(
				"  TrainAcc: " + String.format("% 1.6f", trainingError));
		System.out.println(
				"  ValidAcc: " + String.format("% 1.6f", validationError));
	}

	/*
	 * This method will compute the current cost of a set of weights on a given
	 * data set.
	 */
	public double getAccuracy(ArrayList<double[]> data) {
		Node[] outputLayer = this.outputLayer;
		double[] point;
		double correct = 0.0;
		for (int i = 0; i < data.size(); i++) {
			// for all data points
			point = data.get(i);
			// run point through network
			this.forwardPropagate(point, false);
			int highest = 0; // index of node with highest confidence
			for (int k = 0; k < this.outputLayer.length; k++) {
				highest = (outputLayer[highest].output >= outputLayer[k].output)
						? highest
						: k;
			}
			// index k contains our highest certainty
			if ((int) point[point.length - 1] == highest) {
				correct++;
			}
		}
		return correct / data.size();
	}

	/*
	 * This method will save a derivative calculation made by the backPropagate
	 * method in the nodes as a sum of previous calculations.
	 */
	public void saveCalculation() {
		for (int i = this.network.size() - 1; i > 0; i--) {
			Node[] layer = this.network.get(i);
			Node[] parents = this.network.get(i - 1);
			// want to store delta in node * aj from parent
			for (Node n : layer) {
				// check if bias node is in the parent
				if (i == 1) {
					// parent has bias
					for (int j = 0; j < parents.length; j++) {
						// derivative with respect to weight ij is ai * deltaij
						n.derivatves[j] += n.delta * parents[j].output;
					}
				} else {
					// need to account for extra connection from bias node
					for (int j = 0; j < parents.length + 1; j++) {
						// derivative with respect to weight ij is ai * deltaij
						n.derivatves[j] += (j == 0) ? n.delta
								: n.delta * parents[j - 1].output;
					}
				}
			}
		}
	}

	/*
	 * Zeroes the space in the nodes reserved for derivative sums.
	 */
	public void zeroCalculation() {
		for (int i = 1; i < this.network.size(); i++) {
			Node[] layer = this.network.get(i);
			for (Node n : layer) {
				n.zeroDerivatives();
			}
		}
	}

	/*
	 * This method performs back propagation on a neural network to determine
	 * the partial derivative of the output with respect to any weight.
	 */
	private void backPropUpdate(double[] point, int index) {
		// index is used for verbosity level 4 printing
		if (this.verbosity > 3) {
			System.out.println("    * Forward Propagation on example " + index);
		}
		this.forwardPropagate(point, true);
		// back propagation
		if (this.verbosity > 3) {
			System.out
					.println("    * Backward Propagation on example " + index);
		}
		for (int i = 0; i < this.outputLayer.length; i++) {
			int y = ((int) point[point.length - 1] == i) ? 1 : 0; // class value
			Node n = this.outputLayer[i];
			n.delta = n.activation(n.input) * (1.0 - n.activation(n.input))
					* (-2 * (y - n.output));
		}
		if (this.verbosity > 3) {
			System.out.print("      Layer " + this.network.size()
					+ " (output): Delta_j:");
			for (int j = 0; j < this.outputLayer.length; j++) {
				System.out.print(
						String.format("%7.3f", this.outputLayer[j].delta));
			}
			System.out.println();
		}
		for (int i = this.layers.length; i > 0; i--) {
			Node[] layer = this.network.get(i);
			Node[] children = this.network.get(i + 1);
			for (int j = 0; j < layer.length; j++) {
				Node n = layer[j];
				n.delta = n.activation(n.input) * (1.0 - n.activation(n.input));
				double deltaSum = 0.0;
				for (int k = 0; k < children.length; k++) {
					// weight of parent times child delta. 0th weight is bias
					deltaSum += children[k].weights[j + 1] * children[k].delta;
				}
				n.delta *= deltaSum;
			}
			if (this.verbosity > 3) {
				System.out.print(
						"      Layer " + (i + 1) + " (hidden): Delta_j:");
				for (int j = 0; j < layer.length; j++) {
					System.out.print(String.format("%7.3f", layer[j].delta));
				}
				System.out.println();
			}
		}
	}

	/*
	 * This method checks if the error at any output node is below a precision
	 * threshold for this training set. If all nodes have an error lower than
	 * the given precision, then this method returns true.
	 */
	private boolean errorLimit(double[] point) {
		boolean belowError = true;
		for (int i = 0; i < this.outputLayer.length; i++) {
			// the actual value of the class for this point
			double y = (i == (int) point[point.length - 1]) ? 1.0 : 0.0;
			belowError &= Math.abs(y
					- this.outputLayer[i].output) <= NeuralNetDriver.PRECISION;
		}
		return belowError;
	}

	/*
	 * This method attempts to generate optimal weights for the neural network
	 * to correctly classify data points.
	 */
	public void NeuralNetworkTrain() {
		System.out.println("* Training network (using "
				+ this.trainingSet.size() + " examples)");
		if (this.verbosity > 1) {
			System.out.println("  * Beginning mini-batch gradient descent");
			System.out.println("    (batchSize=" + this.batchSize
					+ ", epochLimit=" + this.epochLimit + ", learningRate="
					+ String.format("%.4f", this.learningRate) + ", lambda="
					+ String.format("%.4f", this.regularization) + ")");
		}
		int numUpdates = 0; // number of weight updates
		int epoch = 0; // current epoch
		int printEpoch = this.epochLimit / 10;
		// absolute errors per output node is less than PRECISION
		boolean belowError = false;
		if (this.verbosity > 2) {
			System.out.println("    Initial model with random weights : Cost = "
					+ String.format("%.6f", getCost(this.trainingSet))
					+ "; Loss = "
					+ String.format("%.6f", getLoss(this.trainingSet))
					+ "; Acc = "
					+ String.format("%.4f", getAccuracy(this.trainingSet)));
		}
		long start = System.currentTimeMillis();
		while (epoch < this.epochLimit && !belowError) {
			if (this.verbosity > 2 && epoch != 0 && epoch % printEpoch == 0) {
				System.out.println("    After " + String.format("%6d", epoch)
						+ " epochs (" + String.format("%6d", numUpdates)
						+ " iter.): Cost = "
						+ String.format("%.6f", getCost(this.trainingSet))
						+ "; Loss = "
						+ String.format("%.6f", getLoss(this.trainingSet))
						+ "; Acc = "
						+ String.format("%.4f", getAccuracy(this.trainingSet)));
			}
			// divide set into mini-batches
			if (this.randomization)
				Collections.shuffle(this.trainingSet); // randomize data points
			belowError = true; // we have not yet made an error this epoch
			if (this.batchSize == 0)
				this.batchSize = this.trainingSet.size();
			for (int b = 0; b < this.trainingSet.size(); b += this.batchSize) {
				// for each mini-batch
				// get the actual size of the batch
				int mySize = (b + this.batchSize <= this.trainingSet.size())
						? this.batchSize
						: this.trainingSet.size() - b;
				// we store a sum of derivatives in the nodes, so clear them
				zeroCalculation();
				for (int example = b; example < b + mySize; example++) {
					// for each example in the batch
					backPropUpdate(this.trainingSet.get(example), b + 1);
					belowError &= errorLimit(this.trainingSet.get(example));
					// add derivative calculation to sum stored in nodes
					saveCalculation();
				}
				// weight update on every edge
				for (int i = 1; i < this.network.size(); i++) {
					Node[] layer = this.network.get(i);
					for (Node n : layer) {
						for (int j = 0; j < n.weights.length; j++) {
							// wnj = wnj - alpha(1/batchSize * derivativej) -
							// 2 * alpha * lambda * wnj
							n.weights[j] = n.weights[j] - this.learningRate
									* (n.derivatves[j] / (double) mySize)
									- 2.0 * this.learningRate
											* this.regularization
											* n.weights[j];
						}
					}
				}
				numUpdates++;
			}
			epoch++;
		}
		long end = System.currentTimeMillis();
		if (this.verbosity > 2) {
			System.out.println("    After " + String.format("%6d", epoch)
					+ " epochs (" + String.format("%6d", numUpdates)
					+ " iter.): Cost = "
					+ String.format("%.6f", getCost(this.trainingSet))
					+ "; Loss = "
					+ String.format("%.6f", getLoss(this.trainingSet))
					+ "; Acc = "
					+ String.format("%.4f", getAccuracy(this.trainingSet)));
		}
		if (this.verbosity > 1) {
			String timePer = String.format("%.4f",
					(double) (end - start) / numUpdates);
			System.out.println("  * Done with fitting!");
			System.out.println("    Training took " + (end - start) + "ms, "
					+ epoch + " epochs, " + numUpdates + " iterations ("
					+ timePer + "ms / iteration)");
			if (epoch >= this.epochLimit) {
				System.out.println("    GD Stop condition: Epoch Limit");
			} else {
				System.out.println("    GD Stop condition: Small Error");
			}
		}
	}

	/*
	 * This method is responsible for setting the weights of a layer given the
	 * layer that immediately precedes it. The parameter containsBias is true
	 * when the parent layer contains the bias node (the parent layer is the
	 * input layer).
	 */
	private void setWeights(Node[] layer, Node[] parents,
			boolean containsBias) {
		for (int i = 0; i < layer.length; i++) {
			// length is all previous layer nodes plus the bias node (if not in
			// the previous layer)
			double[] weights = containsBias ? new double[parents.length]
					: new double[parents.length + 1];
			for (int j = 0; j < weights.length; j++) {
				// init weight to val between -this.weights to this.weights
				// this does not include the max, but it is close enough for our
				// purposes
				weights[j] = -1.0 * this.weights
						+ Math.random() * (2.0 * this.weights);
			}
			layer[i].initWeights(weights);
		}
	}

	/*
	 * This method will construct the feed forward neural network based on the
	 * number of features in the data set and the number of classes.
	 */
	public void createNetwork() {
		System.out.println("* Building network");
		// each data point includes a bias weight and a class
		int len = this.datapoints.get(1).length - 1;
		this.network = new ArrayList<Node[]>();
		// a layer we are adding to the network
		Node[] layer = new Node[len];
		this.network.add(layer);
		// set up input layer
		for (int i = 0; i < len; i++) {
			layer[i] = new Node();
		}
		// set constant output for bias node
		layer[0].output = 1;
		if (this.verbosity > 1) {
			System.out.println("  * Layer sizes (excluding bias neuron(s)):");
			System.out.println("    Layer  1 (input) : "
					+ String.format("%3d", layer.length - 1));
		}
		// set up hidden layers
		for (int i = 0; i < this.layers.length; i++) {
			layer = new Node[this.layers[i]];
			for (int j = 0; j < this.layers[i]; j++) {
				layer[j] = new Node();
			}
			this.setWeights(layer, this.network.get(i), i == 0);
			this.network.add(layer);
			if (this.verbosity > 1) {
				System.out.println("    Layer  " + (i + 2) + " (hidden): "
						+ String.format("%3d", layer.length));
			}
		}
		// set up output layer
		layer = new Node[this.classes];
		for (int i = 0; i < this.classes; i++) {
			layer[i] = new Node();
		}
		this.setWeights(layer, this.network.get(this.layers.length),
				this.layers.length == 0);
		this.network.add(layer);
		this.outputLayer = layer;
		if (this.verbosity > 1) {
			System.out.println("    Layer  " + this.network.size()
					+ " (output): " + String.format("%3d", layer.length));
		}
	}

	/*
	 * This method will scale the feature values in the read data to be within
	 * the range [-1, 1] using min-max normalization on the training set.
	 */
	public void normalize() {
		// we will use indices of 1 through len - 1 to ignore the bias term and
		// the class label
		System.out.println("* Scaling features");
		int len = this.trainingSet.get(0).length;
		double[] mins = new double[len - 2];
		double[] maxes = new double[len - 2];
		// init mins and maxes
		for (int i = 0; i < mins.length; i++) {
			mins[i] = Double.MAX_VALUE;
			maxes[i] = Double.MIN_VALUE;
		}
		double[] diffs = new double[len - 2];
		for (double[] datapoint : this.trainingSet) {
			for (int i = 0; i < len - 2; i++) {
				if (datapoint[i + 1] < mins[i]) {
					mins[i] = datapoint[i + 1];
				}
				if (datapoint[i + 1] > maxes[i]) {
					maxes[i] = datapoint[i + 1];
				}
			}
		}
		for (int i = 0; i < len - 2; i++) {
			diffs[i] = maxes[i] - mins[i];
		}
		// scale the training set
		for (double[] datapoint : this.trainingSet) {
			for (int i = 0; i < len - 2; i++) {
				datapoint[i + 1] = (diffs[i] != 0.0)
						? -1.0 + (2.0 * (datapoint[i + 1] - mins[i]) / diffs[i])
						: -1.0;
			}
		}
		// scale the validation set
		for (double[] datapoint : this.validationSet) {
			for (int i = 0; i < len - 2; i++) {
				datapoint[i + 1] = (diffs[i] != 0.0)
						? -1.0 + (2.0 * (datapoint[i + 1] - mins[i]) / diffs[i])
						: -1.0;
			}
		}
		if (this.verbosity > 1) {
			System.out.println("  * min/max values on training set:");
			for (int i = 0; i < maxes.length; i++) {
				System.out.println("    Feature " + (i + 1) + ": "
						+ String.format("%.3f", mins[i]) + ", "
						+ String.format("%.3f", maxes[i]));
			}
		}
	}

	/*
	 * This method will construct the training and validation sets by using the
	 * first 80% of the data read in as training data, and the rest as
	 * validation data. If the -r flag was specified, the data will be
	 * randomized before splitting.
	 */
	public void splitData() {
		System.out.println("* Doing train/validation split");
		if (this.randomization) {
			Collections.shuffle(this.datapoints);
		}
		int endIndex = (int) (this.datapoints.size() * 0.8);
		this.trainingSet = new ArrayList<double[]>(
				this.datapoints.subList(0, endIndex));
		this.validationSet = new ArrayList<double[]>(
				this.datapoints.subList(endIndex, this.datapoints.size()));
	}

	/*
	 * This method is responsible for reading in training data from the text
	 * file that was provided by the user. This method will convert the given
	 * data to lists of attributes and a class.
	 */
	public boolean readData() {
		String line;
		String[] lineData = { "", "" };
		String[] vals; // numeric values from a line
		try {
			line = this.trainingFile.readLine();
			while (line != null) {
				if (line.length() != 0 && !line.substring(0, 1).equals("#")) {
					// data is of the form (0.665 0.790) (0 1 0)
					lineData = line.split("\\) \\(", 2);
					vals = lineData[0].substring(1).split(" ");
					// add one for bias term and one for class value
					double[] data = new double[vals.length + 2];
					data[0] = 1;
					for (int i = 1; i < vals.length + 1; i++) {
						try {
							data[i] = Double.parseDouble(vals[i - 1]);
						} catch (Exception e) {
							e.printStackTrace();
							System.out.println(
									"Encountered unexpected data in training file:"
											+ vals[i - 1]);
							return false;
						}
					}
					// the class number is the index of "1" divided by 2 to
					// account for spaces
					data[vals.length + 1] = lineData[1].indexOf("1") / 2;
					this.datapoints.add(data);
				}
				line = this.trainingFile.readLine();
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("Error while reading from training file.");
			return false;
		}
		if (this.datapoints.size() == 0 || this.datapoints.get(0).length == 2) {
			System.out.println(
					"No training data was found in the provided file.");
			return false;
		} else {
			// the number of classes is the number of ints that appear in the
			// second paren group
			this.classes = lineData[1].substring(0, lineData[1].length() - 1)
					.split(" ").length;
		}
		try {
			this.trainingFile.close();
		} catch (Exception e) {
			System.out.println("Error closing training file.");
			return false;
		}
		return true;
	}

	/*
	 * This method is responsible for parsing the arguments provided to this
	 * program. If there are missing or invalid arguments, this method will
	 * print an error message and return false
	 */
	public boolean parseArgs(String[] args) {
		String focus;
		for (int i = 0; i < args.length; i++) {
			focus = args[i];
			if (focus.equals("-f")) {
				// training file
				if (i + 1 >= args.length) {
					System.out.println(
							"No training file was provided after \"-f\".");
					return false;
				}
				try {
					this.trainingFile = new BufferedReader(
							new FileReader(args[++i]));
					this.trainingFileName = args[i];
				} catch (Exception e) {
					System.out.println("Unable to open training file.");
					return false;
				}
			} else if (focus.equals("-h")) {
				// hidden layers
				if (i + 1 >= args.length) {
					System.out.println(
							"No hidden layer count was provided after \"-h\".");
					return false;
				}
				try {
					this.layers = new int[Integer.parseInt(args[++i])];
				} catch (Exception e) {
					System.out.println(
							"Hidden layer count should be an integer.");
					return false;
				}
				for (int j = 0; j < this.layers.length; j++) {
					// get the number of nodes per layer
					if (i + 1 + j >= args.length) {
						System.out.println(
								"Not enough layer sizes were provided.");
						return false;
					}
					try {
						this.layers[j] = Integer.parseInt(args[i + 1 + j]);
					} catch (Exception e) {
						System.out.println(
								"Hidden layer node count should be an integer.");
						return false;
					}
				}
				i = i + this.layers.length;
			} else if (focus.equals("-a")) {
				// learning rate
				if (i + 1 >= args.length) {
					System.out.println(
							"No learning rate was provided after \"-a\".");
					return false;
				}
				try {
					this.learningRate = Double.parseDouble(args[++i]);
				} catch (Exception e) {
					System.out.println("Learning rate should be a double.");
					return false;
				}
			} else if (focus.equals("-e")) {
				// epoch limit
				if (i + 1 >= args.length) {
					System.out.println(
							"No epoch limit was provided after \"-e\".");
					return false;
				}
				try {
					this.epochLimit = Integer.parseInt(args[++i]);
				} catch (Exception e) {
					System.out.println("Epoch limit should be an integer.");
					return false;
				}
			} else if (focus.equals("-m")) {
				// batch size
				if (i + 1 >= args.length) {
					System.out.println(
							"No batch size was provided after \"-m\".");
					return false;
				}
				try {
					this.batchSize = Integer.parseInt(args[++i]);
				} catch (Exception e) {
					System.out.println("Batch size should be an integer.");
					return false;
				}
			} else if (focus.equals("-l")) {
				// regularization
				if (i + 1 >= args.length) {
					System.out.println(
							"No regularization parameter was provided after \"-l\".");
					return false;
				}
				try {
					this.regularization = Double.parseDouble(args[++i]);
				} catch (Exception e) {
					System.out.println(
							"Regularization parameter should be a double.");
					return false;
				}
			} else if (focus.equals("-r")) {
				// randomization
				this.randomization = true;
			} else if (focus.equals("-w")) {
				// weight initialization
				if (i + 1 >= args.length) {
					System.out.println(
							"No weight value was provided after \"-w\".");
					return false;
				}
				try {
					this.weights = Double.parseDouble(args[++i]);
				} catch (Exception e) {
					System.out.println("Weight value should be a double.");
					return false;
				}
			} else if (focus.equals("-v")) {
				// verbosity
				if (i + 1 >= args.length) {
					System.out.println(
							"No verbosity level was provided after \"-v\".");
					return false;
				}
				try {
					this.verbosity = Integer.parseInt(args[++i]);
					if (this.verbosity < 1 || this.verbosity > 4) {
						System.out.println(
								"Verbosity level should be between 1 and 4 inclusive.");
						this.verbosity = 1;
					}
				} catch (Exception e) {
					System.out.println("Verbosity level should be an integer.");
					return false;
				}
			} else {
				System.out.println("Invalid argument: " + args[i]);
				return false;
			}
		}
		return true;
	}

	/*
	 * Returns the name of the training file used by this neural network.
	 */
	public String getFileName() {
		return this.trainingFileName;
	}

	public static void main(String[] args) {
		NeuralNetDriver nn = new NeuralNetDriver();
		if (!nn.parseArgs(args)) {
			return;
		}
		if (nn.trainingFile == null) {
			System.out.println("No training file was provided.");
			return;
		}
		nn.datapoints = new ArrayList<double[]>();
		System.out.println("* Reading " + nn.getFileName());
		if (!nn.readData()) {
			return;
		}
		// split data into 80% training and 20% validation
		nn.splitData();
		// normalize our features using training data
		nn.normalize();
		// construct the neural network
		nn.createNetwork();
		// train on training set
		nn.NeuralNetworkTrain();
		// validate using validation set
		nn.evaluate();
	}
}