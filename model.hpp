/*
 Class representing a Feed-Forward, Fully-Connected, Multilayer Neural Network,
 with methods that allow not only to make inferences and evaluate the goodness
 of predictions but also to train the model.
*/


#ifndef MODEL_HPP
#define MODEL_HPP


#include "activation_functions.hpp"
#include "common.hpp"


/**
 Feed-forward, fully-connected, multi-layer neural network for single-output
 regression with settable number of layers, number of neurons each and
 activation functions' kind (either leakyReLU, ReLU or sigmoidal kind),
 equipped with inference, evaluation and training methods, thus supporting both
 forward propagation and backpropagation.
*/
class FullyConnectedNeuralNetwork {
 public:
    explicit FullyConnectedNeuralNetwork(
        std::vector<uint> n_neurons_in_each_layer,
        const std::string& activation_functions);
    float activationFunction(
        const float& input);
    float activationFunctionDerivative(
        const float& input);
    void backPropagation(
        const Tensor1D* target_output,
        const float& learning_rate);
    float computeLossGradientWRTWeight(
        const uint& actual_layer_indx,
        const uint& previous_neuron_indx,
        const uint& current_neuron_indx,
        const Tensor1D* target_output);
    void evaluate(
        const std::vector<Tensor1D*>& validation_samples,
        const std::vector<Tensor1D*>& validation_labels,
        const std::string& output_path,
        const bool& verbose);
    void forwardPropagation(
        const Tensor1D* inputs);
    static Tensor1D squaredErrorLoss(
        const Tensor1D* predicted_output,
        const Tensor1D* target_output);
    static float squaredErrorLossDerivative(
        const float& predicted_output,
        const float& target_output);
    void updateWeightViaSGD(
        const uint& actual_layer_indx,
        const uint& previous_neuron_indx,
        const uint& current_neuron_indx,
        const float& learning_rate,
        const Tensor1D* target_output);
    void train(
        const std::vector<Tensor1D*>& inputs,
        const std::vector<Tensor1D*>& targets,
        const float& learning_rate,
        const uint& n_epochs,
        const bool& verbose);

 private:
    // layers' action potentials, i.e. intermediate linear combination results
    // before activation function application:
    std::vector<Tensor1D*> action_potentials;
    // loss gradients with respect to layers' action potentials:
    std::vector<Tensor1D*> action_potentials_gradients;
    // layers' activations, i.e. results of linear combination with weights
    // followed by activation function application:
    std::vector<Tensor1D*> activations;
    // architecture hyperparameter specifying the kind of activation functions
    // to use:
    std::string activation_functions_kind;
    // inputs, i.e. feature values of the considered sample:
    Tensor1D* inputs;
    // architecture hyperparameters specifying the number of layers and the
    // number of neurons each:
    std::vector<uint> n_neurons_in_each_layer;
    // layers' weights (including biases):
    std::vector<Tensor2D*> weights;
};


#endif
