/*
Bla bla bla.
*/


#include <cmath>


typedef float Scalar;


/**
 Lorem Ipsum.
*/
Scalar sigmoidActivationFunction(Scalar X)
{
    return 1 / (1 + exp(-X));
}

/**
 Lorem Ipsum.
*/
Scalar derivativeOfSigmoidActivationFunction(Scalar X)
{
    return sigmoidActivationFunction(1 - sigmoidActivationFunction(X));
}

/**
 Lorem Ipsum.
*/
class FeedForwardNeuralNetwork
{
    public:
        FeedForwardNeuralNetwork();
        void backPropagation();
        void computeLossGradients();
        void forwardPropagation();
        void updateWeights();

    private:
        int layers;
};

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
