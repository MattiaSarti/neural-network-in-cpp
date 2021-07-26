/*
 Activation functions and their derivatives.
*/


#include "activation_functions.hpp"


/**
 Compute the value of the Leaky ReLU function for the given input.
*/
float leakyReLUFunction(const float& input) {
    return ((input > 0) ? input : (input * 0.1));
}

/**
 Compute the value of the derivative of the Leaky ReLU function for the given
 input.
*/
float leakyReLUFunctionDerivative(const float& input) {
    return ((input > 0) ? static_cast<float>(1) : static_cast<float>(0.1));
}

/**
 Compute the value of the ReLU function for the given input.
*/
float ReLUFunction(const float& input) {
    return std::max(input, static_cast<float>(0));
}

/**
 Compute the value of the derivative of the ReLU function for the given input.
*/
float ReLUFunctionDerivative(const float& input) {
    return ((input > 0) ? static_cast<float>(1) : static_cast<float>(0));
}

/**
 Compute the value of the sigmoid function for the given input.
*/
float sigmoidFunction(const float& input) {
    return 1 / (1 + exp(-input));
}

/**
 Compute the value of the derivative of the sigmoid function for the given
 input.
*/
float sigmoidFunctionDerivative(const float& input) {
    return sigmoidFunction(1 - sigmoidFunction(input));
}
