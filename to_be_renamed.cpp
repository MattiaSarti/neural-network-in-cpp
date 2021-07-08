/*
Bla bla bla.
*/


#include <cmath>
#include <vector>

#include <eigen3/Eigen/Eigen>


using namespace std;


typedef Eigen::MatrixXf     Tensor2D;
typedef Eigen::RowVectorXf  Tensor1D;


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
        FeedForwardNeuralNetwork();
        void backPropagation();
        void computeLossGradients();
        void forwardPropagation();
        void updateWeights();
        void train();

    private:
        // architecture hyperparameters such as number of layers and number
        // of neurons each:
        vector<Tensor1D*> architecture;
        // layers' weights:
        vector<Tensor2D*> weights;
        // intermediate linear combination results of all layers before
        // activation function application, stored for convenience:
        vector<Tensor1D*> action_potentials;
        // loss gradients with respect to the different layers' weights:
        vector<Tensor1D*> gradients;
        float learning_rate;
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

/**
 Lorem Ipsum.
*/
void FeedForwardNeuralNetwork::train()
{
    
}
