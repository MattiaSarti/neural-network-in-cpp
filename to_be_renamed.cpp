/*
Bla bla bla.
*/


#include <cmath>
#include <vector>

#include <eigen3/Eigen/Eigen>


using namespace std;


typedef Eigen::MatrixXf     Tensor2D;
typedef Eigen::RowVectorXf  Tensor1D;
typedef unsigned int        uint;


/**
 Compute the value of the sigmoid function for the given input.
*/
float sigmoidActivationFunction(float X)
{
    return 1 / (1 + exp(-X));
}

/**
 Compute the value of the derivative of the sigmoid function for the given
 input.
*/
float derivativeOfSigmoidActivationFunction(float X)
{
    return sigmoidActivationFunction(1 - sigmoidActivationFunction(X));
}

/**
 Lorem Ipsum.
*/
class FeedForwardNeuralNetwork
{
    public:
        FeedForwardNeuralNetwork(vector<uint> n_neurons_in_each_layer);
        void backPropagation();
        void computeLossGradients();
        void forwardPropagation();
        void updateWeights();
        void train(float learning_rate);

    private:
        // architecture hyperparameters, specifically number of layers and
        // number of neurons each:
        vector<uint> architecture;
        // layers' weights (including biases):
        vector<Tensor2D*> weights;
        // intermediate linear combination results before activation function
        // application of all layers:
        vector<Tensor1D*> action_potentials;
        // activations, i.e. results of linear combination with weights
        // followed by activation function application, of all layers:
        vector<Tensor1D*> activations;
        // loss gradients with respect to the different layers' weights:
        vector<Tensor1D*> gradients;
};

/**
 Lorem Ipsum.
*/
FeedForwardNeuralNetwork::FeedForwardNeuralNetwork(vector<uint> n_neurons_in_each_layer)
{
    architecture = n_neurons_in_each_layer;
    int n_layers = architecture.size();

    // initializing each layer's weights, action potentials, outputs and
    // gradients:
    for (int i = 0; i < n_layers; ++i) {

        // when considering the last layer, the number of outputs has to be
        // considered instead of the number of neurons of the following layer
        // in order to define the connections, i.e. the weights, of the
        // current layer; moreover, biases are not added to the last layer:
        if (i != (n_layers - 1)) {

            //TODO

        } else {

            //TODO

            // adding biases:

        }

        // randomly initializing weights:

        // initializing action potentials and weights' gradients:

    }
}

/**
 Compute the loss value comparing the last layer outputs to the target
 predictions, update the gradient of such loss with respect to each model
 weight and update the latter accordingly, to carry out a single step of
 gradient descent.
*/
void FeedForwardNeuralNetwork::backPropagation()
{
    computeLossGradients();
    updateWeights();
}

/**
 Lorem Ipsum.
*/
void FeedForwardNeuralNetwork::computeLossGradients()
{
    
}

/**
 Lorem Ipsum.
*/
void FeedForwardNeuralNetwork::forwardPropagation()
{
    
}

/**
 Lorem Ipsum.
*/
void FeedForwardNeuralNetwork::updateWeights()
{
    
}

/**
 Lorem Ipsum.
*/
void FeedForwardNeuralNetwork::train(float learning_rate)
{
    
}
