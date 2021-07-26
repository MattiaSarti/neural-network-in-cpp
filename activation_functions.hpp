/*
 Activation functions and their derivatives.
*/


#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP


#include "common.hpp"


/**
 Compute the value of the Leaky ReLU function for the given input.
*/
float leakyReLUFunction(const float& input);

/**
 Compute the value of the derivative of the Leaky ReLU function for the given
 input.
*/
float leakyReLUFunctionDerivative(const float& input);

/**
 Compute the value of the ReLU function for the given input.
*/
float ReLUFunction(const float& input);

/**
 Compute the value of the derivative of the ReLU function for the given input.
*/
float ReLUFunctionDerivative(const float& input);

/**
 Compute the value of the sigmoid function for the given input.
*/
float sigmoidFunction(const float& input);

/**
 Compute the value of the derivative of the sigmoid function for the given
 input.
*/
float sigmoidFunctionDerivative(const float& input);


#endif
