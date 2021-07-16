/*
 Bla bla bla.
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
    return (predicted_output - target_output);
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
        void computeLossGradients(Tensor1D& target_outputs);
        void evaluate(vector<Tensor1D*> validation_samples, vector<Tensor1D*> validation_labels, string output_path, bool verbose);
        void forwardPropagation(Tensor1D& inputs);
        void updateWeightsViaSGD(float learning_rate);
        void train(vector<Tensor1D*> inputs, vector<Tensor1D*> targets, float learning_rate, uint n_epochs);

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
        vector<Tensor2D*> gradients;
};

/**
 Build the neural network architecture components.
*/
FullyConnectedNeuralNetwork::FullyConnectedNeuralNetwork(vector<uint> n_neurons_in_each_layer)        // TODO: understand
{
    // adding the last layer's number of neurons, set to 1 for the
    // sinlge-output regression problem of interest:
    n_neurons_in_each_layer.push_back(1);

    architecture = n_neurons_in_each_layer;
    uint n_layers = n_neurons_in_each_layer.size();

    // making random weight initialization reproducible:
    srand((uint) 0);

    // initializing each layer's weights, action potentials, outputs and
    // gradients:
    for (uint layer_indx = 0; layer_indx < n_layers; ++layer_indx) {

        bool is_not_first_layer = (!(layer_indx == 0));
        bool is_not_last_layer = (!(layer_indx == (n_layers - 1)));
        uint n_neurons_previous_layer;
        uint n_neurons_current_layer = n_neurons_in_each_layer[layer_indx];
        // including a state value for the fictitious bias input for all
        // layers but the output one:
        uint n_states_current_layer = (
            is_not_last_layer ?
            (n_neurons_current_layer + 1) :
            n_neurons_current_layer
        );

        // initializing action potentials:
        action_potentials.push_back(new Tensor1D(n_states_current_layer));

        // declaring layer's activations:// adding biases:
        activations.push_back(new Tensor1D(n_states_current_layer));
        
        if (is_not_first_layer) {
            n_neurons_previous_layer = n_neurons_in_each_layer[layer_indx - 1];
            weights.push_back(new Tensor2D(n_neurons_previous_layer + 1, n_states_current_layer));
            // randomly initializing weights, sampling from a uniform
            // distribution over the [-1;+1] interval:
            weights.back()->setRandom();
        }

        // for all layers but the last one, biases are added as well:
        if (is_not_last_layer) {

            // fictitious, constant (+1) input for adding bias to the
            // following layer:
            action_potentials.back()->coeffRef(n_neurons_current_layer) = 1.0;
            // fictitious, constant (+1) input for adding bias to the
            // following layer:
            activations.back()->coeffRef(n_neurons_current_layer) = 1.0;

            // for all the considered layers that also have weight matrices
            // associated (i.e. discarding the output layer):
            if (is_not_first_layer) {
                // 0, 0, 0, ..., 0, 1:
                weights.back()->col(n_neurons_current_layer).setZero();
                weights.back()->coeffRef(n_neurons_previous_layer, n_neurons_current_layer) = 1.0;
            }

        }

        /*
        // initialing weights' gradients:
        gradients.push_back(new Tensor1D(n_states_current_layer));          // TODO: be coherent
        */

    }

}

/**
 Compute the loss value comparing the last layer outputs to the target
 predictions, update the gradient of such loss with respect to each model
 weight and update the latter accordingly, to carry out a single step of
 gradient descent.
*/
void FullyConnectedNeuralNetwork::backPropagation(Tensor1D& target_outputs, float learning_rate)
{
    computeLossGradients(target_outputs);
    updateWeightsViaSGD(learning_rate);
}

/**
 Lorem Ipsum.
*/
void FullyConnectedNeuralNetwork::computeLossGradients(Tensor1D& target_outputs)
{
    /*
    // computing loss gradients with respect to weights of the last layers:
    (*(gradients.back())) = errorLoss((*(activations.back())), target_outputs);

    // computing loss gradients with respect to weights of the hidden layers,
    // excluding the input layer, which is not associated to any weight matrix
    // - proceeding backwards from the last ones so as to rely on Chain Rule
    // for derivative computation:
    uint last_layer_index = architecture.size() - 2;
    for (uint layer_indx = last_layer_index; layer_indx > 0; --last_layer_index) {
        (*(gradients[layer_indx])) = (*(gradients[layer_indx + 1])) * (weights[layer_indx]->transpose());

        deltas[layer_indx + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[layer_indx + 1]->coeffRef(c)) * neuronLayers[layer_indx]->coeffRef(r);
    }
    */
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
    Tensor1D* cumulative_loss = new Tensor1D(1);
    *cumulative_loss << 0;

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
        // compute the average loss:
        *cumulative_loss += errorLoss(*activations.back(), *validation_labels[sample_indx]);

        // saving the prediction for the current sample to the output file:
        file_stream << *activations.back() << endl;
    }

    // printing the average loss value over the samples of the validation set:
    cout << "average loss on the validation set: " << (*cumulative_loss / n_samples) << endl;

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
 Update the weights of each layer - except the input layer, that is not
 associated to any weight matrix - via Stochastic Gradient Descent based on
 loss gradients with respect to the respective weights, computed previously.
*/
void FullyConnectedNeuralNetwork::updateWeightsViaSGD(float learning_rate)
{
    uint n_actual_layers = architecture.size() - 1;
    uint last_layer_index = architecture.size() - 2;

    // for each layer but the input one, i.e. for each layer with an
    // associated weight matrix (from the first hidden layer to the output
    // layer):
    for (uint layer_indx = 0; layer_indx < n_actual_layers; ++layer_indx) {

        uint n_columns = weights[layer_indx]->cols();
        uint n_rows = weights[layer_indx]->rows();

        // for each row of the current layer weight matrix:
        for (uint row_indx = 0; row_indx < n_rows; ++row_indx) {

            // for all layers but the last one:
            if (layer_indx != last_layer_index) {

                // for each column of the current layer weight matrix but the                         // TODO: understand
                // last one, since biases are not associated to neurons:
                uint n_relevant_columns = n_columns - 1;
                for (uint column_indx = 0; column_indx < n_relevant_columns; ++column_indx) {

                    // updating the weight corresponding to the current row
                    // and column positions:
                    weights[layer_indx]->coeffRef(row_indx, column_indx) += learning_rate * (gradients[layer_indx]->coeffRef(row_indx, column_indx));

                }

            // for the last layer:
            } else {

                // for each column of the current layer weight matrix - biases
                // are not present in the output layer:
                for (uint column_indx = 0; column_indx < n_columns; ++column_indx) {

                    // updating the weight corresponding to the current row
                    // and column positions:
                    weights[layer_indx]->coeffRef(row_indx, column_indx) += learning_rate * (gradients[layer_indx]->coeffRef(row_indx, column_indx));

                }

            }

        }

    }
}

/**
 Train the model on the given samlpes.
*/
void FullyConnectedNeuralNetwork::train(vector<Tensor1D*> training_samples, vector<Tensor1D*> training_labels, float learning_rate, uint n_epochs)
{
    assert(training_samples.size() == training_labels.size());

    uint n_samples = training_samples.size();

    cout << "training started ✓" << endl;

    // for each epoch:
    for (uint epoch_indx = 0; epoch_indx < n_epochs; ++epoch_indx) {

        cout << "epoch " << (epoch_indx + 1) << "/" << n_epochs << endl;

        // for each training sample:
        for (uint sample_indx = 0; sample_indx < n_samples; ++sample_indx) {

            cout << "\t" << "sample features: " << *training_samples[sample_indx] << endl;

            // forward propagation of the sample through the network, computing
            // layers' activations sequentially:
            forwardPropagation(*training_samples[sample_indx]);

            cout << "\t" << "expected output: " << *training_labels[sample_indx] << endl;
            cout << "\t" << "vs" << endl;
            cout << "\t" << "actual output: " << *activations.back() << endl;
            cout << "\t" << "loss value: " << errorLoss(*activations.back(), *training_labels[sample_indx]) << endl;

            // backward propagation of the loss gradients through the network with
            // respect to the different weights eventually updating them
            // accordingly:
            backPropagation(*training_labels[sample_indx], learning_rate);
            cout << "\t" << "weights updated accordingly ✓" << endl;

        }

    }

    cout << "model successfully trained ✓" << endl;

}
