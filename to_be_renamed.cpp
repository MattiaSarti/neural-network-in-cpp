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
        void backPropagation(Tensor1D& target_outputs);
        void computeLossGradients(Tensor1D& target_outputs);
        void forwardPropagation(Tensor1D& inputs);
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
    int n_layers = n_neurons_in_each_layer.size();

    // initializing each layer's weights, action potentials, outputs and
    // gradients:
    for (uint i = 0; i < n_layers; ++i) {

        // :
        uint n_neurons = (
            (i != (n_layers - 1)) ?
            n_neurons_in_each_layer[i + 1] :
            n_neurons_in_each_layer[i]
        );

        // initializing action potentials:
        action_potentials.push_back(new Tensor1D(n_neurons));

        // declaring layer's activations:// adding biases:
        activations.push_back(new Tensor1D());
        
        if (i != 0) {
            // randomly initializing weights:
            weights.push_back(new Tensor2D());
            weights.back();
        }

        // when considering the last
        // layer, the number of outputs has to be considered instead of the
        // number of neurons of the following layer in order to define the
        // connections, i.e. the weights, of the current layer; moreover,
        // biases are not added to the last layer:
        if (i != (n_layers - 1)) {

            // :
            action_potentials.back();

            // :
            activations.back();

            // :
            if (i != 0) {
                weights.back();
                weights.back();
            }

        } else {                                                   // TODO: use "insert" in place of "coeffRef" for Tensor2D

        }

        // initialing weights' gradients:
        gradients.push_back(new Tensor1D(n_neurons));

    }

}

/**
 Compute the loss value comparing the last layer outputs to the target
 predictions, update the gradient of such loss with respect to each model
 weight and update the latter accordingly, to carry out a single step of
 gradient descent.
*/
void FeedForwardNeuralNetwork::backPropagation(Tensor1D& target_outputs)
{
    computeLossGradients(target_outputs);
    updateWeights();
}

/**
 Lorem Ipsum.
*/
void FeedForwardNeuralNetwork::computeLossGradients(Tensor1D& target_outputs)
{
    // TODO: compute actual gradients, not just errors
    (*(gradients.back())) = target_outputs - (*(activations.back()));

}

/**
 Lorem Ipsum.
*/
void FeedForwardNeuralNetwork::forwardPropagation(Tensor1D& inputs)
{
    int n_layers = architecture.size();

    // setting the input layer values as the inputs - accessing a block of
    // size (p,q) starting at (i,j) via ".block(i,j,p,q)" for tensors:
    uint n_input_neurons = activations.front()->size();  // TODO: understand why not " = architecture[0];"
    activations.front()->block(0, 0, 1, n_input_neurons - 1) = inputs;

    // propagating the inputs to the outputs by computing the activations of
    // each neuron in each layer from the previous layer's activations by
    // linearly combining the neuron inputs with the respective weights first
    // and then applying the activation function, layer by layer:
    for (int i = 1; i < n_layers; ++i) {

        // computing the result of the linear combination with weights (i.e.
        // the action potential):
        (*activations[i]) = (*activations[i - 1]) * (*weights[i - 1]);

        // for all layers but the last one, whose outputs require a different
        // activation function:
        if (i != n_layers - 1) {
            // applying the activation function to the linear combination
            // result:
            uint n_neurons = architecture[i];
            activations[i]->block(0, 0, 1, n_neurons).unaryExpr(
                ptr_fun(sigmoidActivationFunction)
            );
        }

    }
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
