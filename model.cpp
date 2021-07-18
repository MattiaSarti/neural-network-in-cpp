/*
 Definition of the class representing a Feed-Forward, Fully-Connected,
 Multilayer Neural Network, with methods that allow not only to make
 inferences and evaluate the goodness of predictions but also to train the
 model.
*/


#include "common.hpp"//"model.hpp"  // FIXME


using namespace std;


/**
 Compute the value of the simple difference loss (i.e. signed error) between
 the predicted output and the expected, target output.
 NOTE: the derivative of such loss with respect to the predicted output is
 unitary and simplifies computations.
*/
Tensor1D errorLoss(Tensor1D& predicted_output, Tensor1D& target_output)
{
    return (target_output - predicted_output);
}

/**
 Compute the value of the sigmoid function for the given input.
*/
float sigmoidActivationFunction(float input)
{
    return 1 / (1 + exp(-input));
}

/**
 Compute the value of the derivative of the sigmoid function for the given
 input.
*/
float derivativeOfSigmoidActivationFunction(float input)
{
    return sigmoidActivationFunction(1 - sigmoidActivationFunction(input));
}

/**
 Feed-forward, fully-connected, multi-layer neural network for single-output
 regression with sigmoidal activation functions and biases in hidden layers.
*/
class FullyConnectedNeuralNetwork
{
    public:
        FullyConnectedNeuralNetwork(vector<uint> n_neurons_in_each_layer);
        void backPropagation(Tensor1D& target_outputs, float learning_rate);
        float computeLossGradientWRTWeight(uint layer_indx, uint row_indx, uint column_indx);
        void evaluate(vector<Tensor1D*> validation_samples, vector<Tensor1D*> validation_labels, string output_path, bool verbose);
        void forwardPropagation(Tensor1D& inputs);
        void updateWeightViaSGD(uint layer_indx, uint row_indx, uint column_indx, float learning_rate);
        void train(vector<Tensor1D*> inputs, vector<Tensor1D*> targets, float learning_rate, uint n_epochs, bool verbose);

    private:
        // architecture hyperparameters, specifically number of layers and
        // number of neurons each:
        vector<uint> architecture;
        // layers' weights (including biases):
        vector<Tensor2D*> weights;
        // layers' action potentials, i.e. intermediate linear combination
        // results before activation function application:
        vector<Tensor1D*> action_potentials;
        // layers' activations, i.e. results of linear combination with weights
        // followed by activation function application:
        vector<Tensor1D*> activations;
        // errors on layers' activations:
        vector<Tensor1D*> activation_errors;
};

/**
 Build the neural network architecture components.
*/
FullyConnectedNeuralNetwork::FullyConnectedNeuralNetwork(vector<uint> n_neurons_in_each_layer)
{
    // adding the last layer's number of neurons, set to 1 for the
    // sinlge-output regression problem of interest:
    n_neurons_in_each_layer.push_back(1);

    architecture = n_neurons_in_each_layer;
    uint n_layers = n_neurons_in_each_layer.size();

    // making random weight initialization reproducible:
    srand((uint) 0);

    // initializing each layer's weights, action potentials, activations (i.e.
    // outputs) and activations' errors:
    for (uint layer_indx = 0; layer_indx < n_layers; ++layer_indx) {

        bool is_not_first_layer = (!(layer_indx == 0));
        bool is_not_last_layer = (!(layer_indx == (n_layers - 1)));
        uint n_actual_neurons_previous_layer;
        uint n_actual_neurons_current_layer = n_neurons_in_each_layer[layer_indx];
        // including an additional neuron for the fictitious bias input for
        // all layers but the output one, which is not subject to bias
        // summation during the action potential computation:
        uint n_neurons_current_layer = (
            is_not_last_layer ?
            (n_actual_neurons_current_layer + 1) :
            n_actual_neurons_current_layer
        );

        // declaring the current layer's action potentials:
        action_potentials.push_back(new Tensor1D(n_neurons_current_layer));

        // declaring the current layer's activations:
        activations.push_back(new Tensor1D(n_neurons_current_layer));

        // declaring errors on the considered layer's activations:
        activation_errors.push_back(new Tensor1D(n_neurons_current_layer));
        
        // weight matrices are associated to each layer but one (they are
        // in-between consecutive layers, connecting their neurons), so they
        // are defined for each layer but the input one referring to the
        // previous layer as well:
        if (is_not_first_layer) {
            n_actual_neurons_previous_layer = n_neurons_in_each_layer[layer_indx - 1];
            // declaring the weights that connect the previous layer to the
            // considered one - the number of neurons in the previous layer is
            // extended considering an additional, fictitious input to
            // consider the bias term in the weight matrix itself:
            weights.push_back(new Tensor2D(n_actual_neurons_previous_layer + 1, n_neurons_current_layer));
            // randomly initializing weights, sampling from a uniform
            // distribution over the [-1;+1] interval - NOTE: this
            // initialization is overwritten for biases later in the
            // constructor:
            weights.back()->setRandom();
        }

        // for all layers but the last one, biases are considered as well:
        if (is_not_last_layer) {

            // fictitious, constant (+1) input for adding bias to the            // TODO: understand: why also on action potentials?
            // following layer:
            action_potentials.back()->coeffRef(n_neurons_current_layer - 1) = 1.0;
            // fictitious, constant (+1) input for adding bias to the            // TODO: understand: why also on action potentials?
            // following layer:
            activations.back()->coeffRef(n_neurons_current_layer - 1) = 1.0;

            // for all the layers (among the considered ones, where biases are
            // introduced) that also have weight matrices associated (i.e.
            // discarding the input layer with the aforementioned convention):
            if (is_not_first_layer) {
                // initializg bias terms with zeros (overwriting the previous,
                // general initialization, that was meant for the actual
                // weights) by setting the last columns of the weight matrix,
                // i.e. the weights that connect the fictitious bias inputs of
                // the previous layer to the current layer's neurons, to all
                // zeros, as appropriate for biases:
                weights.back()->col(n_neurons_current_layer - 1).setZero();
                // the bias term connecting the fictitious neuron of the
                // previous layer to the fictitious neuron of the current
                // layer is set to 1:                                             // TODO: understand: why? is it ever used?
                weights.back()->coeffRef(n_actual_neurons_previous_layer, n_neurons_current_layer - 1) = 1.0;
            }

        }

    }

}

/**
 Compute the loss value comparing the last layer outputs to the target
 predictions, compute the gradient of such loss with respect to each model
 weight and update these latter accordingly, to carry out a single step of
 gradient descent.
 The weights of each layer - except the input layer, that is not
 associated to any weight matrix - are updated via Stochastic Gradient Descent,
 with mini-batched of size 1.
*/
void FullyConnectedNeuralNetwork::backPropagation(Tensor1D& target_outputs, float learning_rate)
{
    uint last_hidden_layer_index = architecture.size() - 2;

    // computing the error on the model outputs, i.e. on the activations of
    // the last layer, so as to start backpropagation, propagating it through
    // previous layers in turn, backwards:
    (*(activation_errors.back())) = errorLoss((*(activations.back())), target_outputs);

    // for each layer but the output one, i.e. for each layer with a layer
    // ahead to backpropagate errors from (from the input layer to the last
    // hidden layer) - NOTE: weight matrices can be enumerated with these
    // layers and so are they (considering the layer they "start from" as the
    // layer they are associated with):
    for (int layer_indx = last_hidden_layer_index; layer_indx >= 0; --layer_indx) {

        uint n_columns = weights[layer_indx]->cols();
        uint n_rows = weights[layer_indx]->rows();

        // computing the errors on the activation functions of the current
        // layer based on the (already computed) errors on the activation
        // functions of the following layer - this is not necessary for the
        // input layer, which is skipped:
        if (layer_indx > 0) {
            (*(activation_errors[layer_indx])) = (*(activation_errors[layer_indx + 1])) * (weights[layer_indx]->transpose());
        }

        // for each row of the current layer weight matrix, i.e. for each
        // input neuron of the considered layer:
        for (uint row_indx = 0; row_indx < n_rows; ++row_indx) {

            // for the input layer and all the hidden layers but the last
            // hidden one:
            if (layer_indx != last_hidden_layer_index) {

                // for each column of the current layer weight matrix, i.e.
                // for each neuron of the following layer, but the last one
                // since biases are not associated to neurons?!:                        // TODO: understand
                uint n_relevant_columns = n_columns - 1;
                for (uint column_indx = 0; column_indx < n_relevant_columns; ++column_indx) {

                    // updating the weight corresponding to the current row
                    // and column positions, i.e. the weight representing the
                    // connection from the considered neuron of the current
                    // layer to the considered neuron of the following layer:
                    updateWeightViaSGD(layer_indx, row_indx, column_indx, learning_rate);

                }

            // for the last hidden layer:
            } else {

                // for each column of the current layer weight matrix, i.e.
                // for each neuron of the following layer - biases are not
                // present in the output layer?!:                                       // TODO: understand
                for (uint column_indx = 0; column_indx < n_columns; ++column_indx) {

                    // updating the weight corresponding to the current row
                    // and column positions, i.e. the weight representing the
                    // connection from the considered neuron of the current
                    // layer to the considered neuron of the following layer:
                    updateWeightViaSGD(layer_indx, row_indx, column_indx, learning_rate);

                }

            }

        }

    }
}

/**
 Compute the loss gradient with respect to the selected weight, which is
 individuated by its (starting, not ending) layer index and its position in
 such layer weight matrix.
*/
float FullyConnectedNeuralNetwork::computeLossGradientWRTWeight(uint layer_indx, uint row_indx, uint column_indx)
{
    return activation_errors[layer_indx + 1]->coeffRef(column_indx) * derivativeOfSigmoidActivationFunction(action_potentials[layer_indx + 1]->coeffRef(column_indx)) * activations[layer_indx]->coeffRef(row_indx);
}

/**
 Evaluate the model on the validation set, printing the average loss value and
 saving predictions to the file system.
*/
void FullyConnectedNeuralNetwork::evaluate(vector<Tensor1D*> validation_samples, vector<Tensor1D*> validation_labels, string output_path, bool verbose = false)
{
    assert(validation_samples.size() == validation_labels.size());

    string space = "       ";
    ofstream file_stream(output_path);
    float cumulative_loss = 0;

    if (verbose) {
        cout << "evaluating predictions for every validation sample:" << endl;
    }

    // for each validation sample:
    uint n_samples = validation_samples.size();
    for (uint sample_indx = 0; sample_indx < n_samples; ++sample_indx) {

        // forward propagation of the sample through the network, computing
        // layers' activations sequentially:
        forwardPropagation(*validation_samples[sample_indx]);

        if (verbose) {
            cout << space;
            cout << "expected output: " << *validation_labels[sample_indx];
            cout << space << "vs" << space;
            cout << "actual output: " << *activations.back();
            cout << space << "->" << space;
            cout << "loss value: " << errorLoss(*activations.back(), *validation_labels[sample_indx]);
            cout << endl;
        }

        // cumulating the loss value for the current sample to eventually
        // compute the average loss - MAE (Mean Absolute Error) employed:
        cumulative_loss += abs(errorLoss(*activations.back(), *validation_labels[sample_indx]).value());

        // saving the prediction for the current sample to the output file:
        file_stream << *activations.back() << endl;
    }

    // printing the average loss value over the samples of the validation set:
    cout << "average loss on the validation set: " << (cumulative_loss / n_samples) << endl;

    file_stream.close();

}

/**
 Forward-propagate the input tensor through the model layers, computing the
 output.
*/
void FullyConnectedNeuralNetwork::forwardPropagation(Tensor1D& inputs)
{
    int n_layers = architecture.size();

    // setting the input layer values as the inputs - accessing a block of
    // size (p,q) starting at (i,j) via ".block(i,j,p,q)" for tensors:
    uint n_input_neurons = activations.front()->size();
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
 Update the selected weight, which is individuated by its (starting, not
 ending) layer index and its position in such layer weight matrix, to carry
 out a step of Stochastic Gradient Descent modulated by the given learning
 rate.
*/
void FullyConnectedNeuralNetwork::updateWeightViaSGD(uint layer_indx, uint row_indx, uint column_indx, float learning_rate)
{
    weights[layer_indx]->coeffRef(row_indx, column_indx) += learning_rate * computeLossGradientWRTWeight(layer_indx, row_indx, column_indx);
}

/**
 Train the model on the given samlpes.
*/
void FullyConnectedNeuralNetwork::train(vector<Tensor1D*> training_samples, vector<Tensor1D*> training_labels, float learning_rate, uint n_epochs, bool verbose = false)
{
    assert(training_samples.size() == training_labels.size());

    uint n_samples = training_samples.size();

    cout << "training started ✓" << endl;

    // for each epoch:
    for (uint epoch_indx = 0; epoch_indx < n_epochs; ++epoch_indx) {

        cout << "epoch " << (epoch_indx + 1) << "/" << n_epochs << endl;

        // for each training sample:
        for (uint sample_indx = 0; sample_indx < n_samples; ++sample_indx) {

            // forward propagation of the sample through the network, computing
            // layers' activations sequentially:
            forwardPropagation(*training_samples[sample_indx]);

            // backward propagation of the loss gradients through the network with
            // respect to the different weights eventually updating them
            // accordingly:
            backPropagation(*training_labels[sample_indx], learning_rate);

            if (verbose) {
                cout << "\t" << "sample features: " << *training_samples[sample_indx] << endl;
                cout << "\t" << "expected output: " << *training_labels[sample_indx] << endl;
                cout << "\t" << "vs" << endl;
                cout << "\t" << "actual output: " << *activations.back() << endl;
                cout << "\t" << "loss value: " << errorLoss(*activations.back(), *training_labels[sample_indx]) << endl;
                cout << "\t" << "weights updated accordingly ✓" << endl;
            }

        }

    }

    cout << "model successfully trained ✓" << endl;

}
