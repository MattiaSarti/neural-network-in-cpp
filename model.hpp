/*
 Model declaration.
*/


#ifndef MODEL_HPP
#define MODEL_HPP


#include "common.hpp"


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
        void evaluate(vector<Tensor1D*> validation_samples, vector<Tensor1D*> validation_labels);
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


#endif
