/*
 Definition of the class representing a Feed-Forward, Fully-Connected,
 Multilayer Neural Network, with methods that allow not only to make
 inferences and evaluate the goodness of predictions but also to train the
 model.
*/


#include "common.hpp"//"model.hpp"  // FIXME


/**
 Compute the value of the Leaky ReLU function for the given input.
*/
float leakyReLUFunction(const float& input) {
    return std::max(input, static_cast<float>(0));  // TODO(me)
}

/**
 Compute the value of the derivative of the Leaky ReLU function for the given
 input.
*/
float leakyReLUFunctionDerivative(const float& input) {
    return ((input > 0) ? static_cast<float>(1) : static_cast<float>(0));  // TODO(me)
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

/**
 Feed-forward, fully-connected, multi-layer neural network for single-output
 regression with settable kind of activation functions (either leakyReLU, ReLU
 or sigmoidal kind) and with biases in hidden layers.
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

/**
 Build the neural network architecture components.
*/
FullyConnectedNeuralNetwork::FullyConnectedNeuralNetwork(
    std::vector<uint> n_neurons_in_each_layer,
    const std::string& activation_functions
) {
    // asserting that a valid activation functions' kind is given:
    assert((activation_functions == "leakyReLU")
            || (activation_functions == "ReLU")
            || (activation_functions == "sigmoid"));

    // setting the kind of activation functions employed:
    this->activation_functions_kind = activation_functions;

    // adding the last layer's number of neurons, set to 1 for the
    // sinlge-output regression problem of interest:
    n_neurons_in_each_layer.push_back(1);

    this->n_neurons_in_each_layer = n_neurons_in_each_layer;
    uint n_layers = n_neurons_in_each_layer.size();

    // making random weight initialization reproducible:
    srand(static_cast<uint>(0));  // NOLINT(cert-msc32-c, cert-msc51-cpp)

    // considering the input features as the input layer's values - and adding
    // a fictitious, constant (+1) input for adding bias to the following
    // layer's action potentials:
    this->inputs = new Tensor1D(n_neurons_in_each_layer[0] + 1);
    this->inputs->coeffRef(n_neurons_in_each_layer[0]) = 1.0;

    // initializing each layer's weights, action potentials, activations (i.e.
    // outputs) and gradients of the latter two:
    for (uint layer_indx = 1; layer_indx < n_layers; ++layer_indx) {
        bool is_not_last_layer = (!(layer_indx == (n_layers - 1)));
        uint n_actual_neurons_current_layer = n_neurons_in_each_layer[
            layer_indx];
        // including an additional neuron whose activation is the fictitious
        // bias input (to consider it as a weight) for the action potential
        // computation for the next layer of the considered one for all the
        // layers but the output one, which does not need one:
        uint n_neurons_current_layer = (is_not_last_layer
            ? (n_actual_neurons_current_layer + 1)
            : n_actual_neurons_current_layer);
        uint n_neurons_previous_layer = n_neurons_in_each_layer[
            layer_indx - 1] + 1;

        // declaring the current layer's action potentials and their loss
        // gradients:
        this->action_potentials.push_back(
            new Tensor1D(n_actual_neurons_current_layer));
        this->action_potentials_gradients.push_back(
            new Tensor1D(n_actual_neurons_current_layer));

        // declaring the current layer's activations:
        this->activations.push_back(
            new Tensor1D(n_neurons_current_layer));

        // declaring the weight matrix of the current layer, the weights that
        // connect the previous layer to the considered one - weight matrices
        // are associated to each layer but one (they are in-between
        // consecutive layers, connecting their neurons), so they are defined
        // for each layer but the input one as referring to the previous layer
        // as well - the number of neurons in the previous layer is
        // extended taking into account the additional, fictitious input to
        // consider the bias term in the weight matrix itself:
        this->weights.push_back(
            new Tensor2D(n_neurons_previous_layer,
                         n_actual_neurons_current_layer));
        // randomly initializing weights, sampling from a uniform distribution
        // over the [-1;+1] interval - this initialization is overwritten for
        // biases later in the constructor:
        this->weights.back()->setRandom();

        // for all layers but the last one, biases are considered as additional
        // neurons as well:
        if (is_not_last_layer) {
            // fictitious, constant (+1) activation as input for adding bias to
            // the following layer's action potentials:
            this->activations.back()
                ->coeffRef(n_neurons_current_layer - 1) = 1.0;
            // initializing bias terms with zeros (overwriting the previous,
            // general initialization, that was meant for the actual
            // weights) by setting the last columns of the weight matrix,
            // i.e. the weights that connect the fictitious bias inputs of
            // the previous layer to the current layer's neurons, to all
            // zeros, as appropriate for biases:
            this->weights.back()->row(n_neurons_previous_layer - 1).setZero();
        }
    }
}

/**
 Compute the activation function value for the given input, with the kind of
 activation function employed determined by the set architectural choice:
*/
float FullyConnectedNeuralNetwork::activationFunction(
    const float& input
) {
    if (this->activation_functions_kind == "leakyReLU") {
        return leakyReLUFunction(input);
    } else if (this->activation_functions_kind == "ReLU") {
        return ReLUFunction(input);
    } else if (this->activation_functions_kind == "sigmoid") {
        return sigmoidFunction(input);
    } else {
        throw std::invalid_argument("Ill-designed code.");
    }
}

/**
 Compute the derivative of the activation function value for the given input,
 with the kind of activation function employed, and so of its derivative,
 determined by the set architectural choice:
*/
float FullyConnectedNeuralNetwork::activationFunctionDerivative(
    const float& input
) {
    if (this->activation_functions_kind == "leakyReLU") {
        return leakyReLUFunctionDerivative(input);
    } else if (this->activation_functions_kind == "ReLU") {
        return ReLUFunctionDerivative(input);
    } else if (this->activation_functions_kind == "sigmoid") {
        return sigmoidFunctionDerivative(input);
    } else {
        throw std::invalid_argument("Ill-designed code.");
    }
}

/**
 Compute the loss value comparing the last layer outputs to the target
 predictions, compute the gradient of such loss with respect to each model
 weight and update these latter accordingly, to carry out a single step of
 gradient descent; the weights of each layer - except the input layer, that is not
 associated to any weight matrix - are updated via Stochastic Gradient Descent,
 with mini-batched of size 1.
*/
void FullyConnectedNeuralNetwork::backPropagation(
    const Tensor1D* target_output,
    const float& learning_rate
) {
    // for each actual layer (all layers but the input one), i.e. for each
    // layer with a weight matrix associated:
    uint last_actual_layer_index = this->n_neurons_in_each_layer.size() - 2;
    for (int actual_layer_indx = last_actual_layer_index;
            actual_layer_indx >= 0; --actual_layer_indx
    ) {
        uint n_neurons_previous_layer = this->weights[actual_layer_indx]
            ->rows();
        uint n_neurons_current_layer = this->weights[actual_layer_indx]
            ->cols();

        // for each row of the current layer weight matrix, i.e. for each
        // input neuron to the considered layer:
        for (uint previous_neuron_indx = 0; previous_neuron_indx
                < n_neurons_previous_layer; ++previous_neuron_indx
        ) {
            // for each column of the current layer weight matrix, i.e.
            // for each neuron of the considered layer:
            for (uint current_neuron_indx = 0; current_neuron_indx
                    < n_neurons_current_layer; ++current_neuron_indx
            ) {
                // updating the weight connecting the two considered neurons:
                this->updateWeightViaSGD(actual_layer_indx,
                    previous_neuron_indx, current_neuron_indx, learning_rate,
                    target_output);
            }
        }
    }
}

/**
 Compute the loss gradient with respect to the selected weight, which is
 individuated by its (starting, not ending) layer index and its position in
 such layer weight matrix.
*/
float FullyConnectedNeuralNetwork::computeLossGradientWRTWeight(
    const uint& actual_layer_indx,
    const uint& previous_neuron_indx,
    const uint& current_neuron_indx,
    const Tensor1D* target_output
) {
    /*
    NOTE - the mathematical formulation of Chain Rule follows:

    d/dw_n(l) = d/dw_n(p_n) * d/dp_n(l) =
              = d/dw_n(p_n) * d/dp_n(a_n) * d/da_n(l) =
              = d/dw_n(p_n) * 1 * squaredErrorLossDerivative =
              = a_n-1 * 1 * squaredErrorLossDerivative    in general

            ( = 1 * 1 * squaredErrorLossDerivative    when w_n = bias)

        for the output layer (n == N),

    d/dw_n(l) = d/dw_n(p_n) * d/dp_n(l) =
              = d/dw_n(p_n) * d/dp_n(a_n) * d/da_n(l) =
              = d/dw_n(p_n) * activationFunctionDerivative * d/da_n(l) =
              = a_n-1 * activationFunctionDerivative *
                * ⨊_i(d/dp_n+1_i(l) * d/da_n(p_n+1_i))
              = a_n-1 * activationFunctionDerivative *
                * ⨊_i(d/dp_n+1_i(l) * w_n+1_i)    in general

            ( = 1 * activationFunctionDerivative *
                * ⨊_i(d/dp_n+1_i(l) * w_n+1_i)    when w_n = bias)

        for the hidden layers (N < n < 0),

    where  a = activation of n-th layer,
           l = loss,
           p = action potential of n-th layer,
           w_n = weight of n-th layer                                   ❏
    */
    // keeping track of the loss derivative with respect to the action
    // potential of the current neuron in the considered layer so as to
    // propagate backwards such gradient through the previous layers, later, as
    // a starting point to compute the gradients of the previous layer's
    // weights via Chain Rule:
    if (actual_layer_indx != this->n_neurons_in_each_layer.size() - 2) {
        // for the hidden layers:
        float activation_derivative = 0;

        for (uint following_neuron_indx = 0; following_neuron_indx <
                this->n_neurons_in_each_layer[actual_layer_indx + 2];
                ++following_neuron_indx
        ) {
            activation_derivative += this
                ->action_potentials_gradients[actual_layer_indx + 1]
                    ->coeffRef(following_neuron_indx)
                * this->weights[actual_layer_indx + 1]
                    ->coeffRef(actual_layer_indx, following_neuron_indx);
        }

        this->action_potentials_gradients[actual_layer_indx]
            ->coeffRef(current_neuron_indx) = activation_derivative
                * this->activationFunctionDerivative(
                    this->action_potentials[actual_layer_indx]
                        ->coeffRef(current_neuron_indx));
    } else {
        // for the output layer:
        this->action_potentials_gradients[actual_layer_indx]
            ->coeffRef(current_neuron_indx) = this->squaredErrorLossDerivative(
                this->action_potentials[actual_layer_indx]
                    ->coeffRef(current_neuron_indx),
                target_output->coeffRef(0));
    }

    // computing the loss gradient with respect to the considered weight of
    // the current layer:
    float previous_activation;
    if (actual_layer_indx != 0) {
        // when the previous layer is not the input layer:
        previous_activation = this->activations[actual_layer_indx - 1]
            ->coeffRef(previous_neuron_indx);
    } else {
        // when the previous layer is the input layer:
        previous_activation = this->inputs->coeffRef(previous_neuron_indx);
    }
    float loss_gradient_wrt_weight = previous_activation
        * this->action_potentials_gradients[actual_layer_indx]
            ->coeffRef(current_neuron_indx);

    return loss_gradient_wrt_weight;
}

/**
 Evaluate the model on the validation set, printing the average loss value and
 saving predictions to the file system.
*/
void FullyConnectedNeuralNetwork::evaluate(
    const std::vector<Tensor1D*>& validation_samples,
    const std::vector<Tensor1D*>& validation_labels,
    const std::string& output_path,
    const bool& verbose
) {
    assert(validation_samples.size() == validation_labels.size());

    std::string space = "       ";
    std::ofstream file_stream(output_path);
    long double cumulative_loss = 0;

    if (verbose) {
        std::cout << "evaluating predictions for every validation sample:"
            << std::endl;
    }

    // for each validation sample:
    uint n_samples = validation_samples.size();
    for (uint sample_indx = 0; sample_indx < n_samples; ++sample_indx) {
        // forward propagation of the sample through the network, computing
        // layers' activations sequentially:
        this->forwardPropagation(validation_samples[sample_indx]);

        if (verbose) {
            std::cout << space;
            std::cout << "expected output: "
                << *validation_labels[sample_indx];
            std::cout << space << "vs" << space;
            std::cout << "actual output: " << *(this->activations.back());
            std::cout << space << "->" << space;
            std::cout << "loss value: " << squaredErrorLoss(
                this->activations.back(), validation_labels[sample_indx]);
            std::cout << std::endl;
        }

        // cumulating the loss value for the current sample to eventually
        // compute the average loss - MAE (Mean Absolute Error) employed:
        cumulative_loss += abs(
            squaredErrorLoss(this->activations.back(),
                             validation_labels[sample_indx])
            .value());

        // saving the prediction for the current sample to the output file:
        file_stream << *(this->activations.back()) << std::endl;
    }

    // printing the average loss value over the samples of the validation set:
    std::cout << "average loss on the validation set: "
        << (cumulative_loss / n_samples) << std::endl;

    file_stream.close();
}

/**
 Forward-propagate the input tensor through the model layers, computing the
 output.
*/
void FullyConnectedNeuralNetwork::forwardPropagation(
    const Tensor1D* inputs
) {
    // number of layers without considering the input one:
    int n_actual_layers = this->n_neurons_in_each_layer.size() - 1;

    // setting the input layer values of all the input neurons but the last,
    // fictitious one representing the constant bias multiplier (+1) as the
    // inputs for - accessing a block of size (p, q) starting at (i, j) via
    // ".block(i, j, p, q)" for tensors:
    uint n_actual_neurons = this->n_neurons_in_each_layer[0];
    this->inputs->block(0, 0, 1, n_actual_neurons) = *inputs;

    // propagating the inputs to the output by computing the activations of
    // each neuron in each layer from the previous layer's activations by
    // linearly combining the neuron inputs with the respective weights first
    // and then applying the activation function, layer by layer:
    for (uint actual_layer_indx = 0; actual_layer_indx < n_actual_layers; ++actual_layer_indx) {
        n_actual_neurons = this->n_neurons_in_each_layer[actual_layer_indx + 1];

        // computing the result of the linear combination of the previous
        // layer's activations (or of the inputs, in case the input layer is
        // the previous one) with weights, i.e. the action potential:
        if (actual_layer_indx != 0) {
            (*(this->action_potentials[actual_layer_indx])) = (*(this->
                    activations[actual_layer_indx - 1]))
                * (*(this->weights[actual_layer_indx]));
        } else {
            (*(this->action_potentials[actual_layer_indx])) = (*(this->
                inputs)) * (*(this->weights[actual_layer_indx]));
        }

        // usign the action potentials as starting values for the activations:
        (this->activations[actual_layer_indx])
            ->block(0, 0, 1, n_actual_neurons) = (*(this->action_potentials[
                actual_layer_indx]));

        // for all layers but the last one, whose output does not require an
        // activation function:
        if (actual_layer_indx != (n_actual_layers - 1)) {
            // applying the activation function to the linear combination
            // results (not to the activation representing the fictitious,
            // constant bias multiplier for the following layer (+1)):
            this->activations[actual_layer_indx]
                ->block(0, 0, 1, n_actual_neurons) = this
                    ->activations[actual_layer_indx]
                        ->block(0, 0, 1, n_actual_neurons).unaryExpr(std::bind(
                            &FullyConnectedNeuralNetwork::activationFunction,
                            this,
                            std::placeholders::_1));
        }
    }
}

/**
 Compute the value of the squared difference loss (i.e. squared error) between
 the predicted output and the expected, target output.
*/
Tensor1D FullyConnectedNeuralNetwork::squaredErrorLoss(
    const Tensor1D* predicted_output,
    const Tensor1D* target_output
) {
    return (*target_output - *predicted_output).array().pow(2);
}

/**
 Compute the derivative value of the squared difference loss (i.e. squared
 error) between the predicted output and the expected, target output with
 respect to the predicted output.
*/
float FullyConnectedNeuralNetwork::squaredErrorLossDerivative(
    const float& predicted_output,
    const float& target_output
) {
    /*
    NOTE - the mathematical derivation follows:

    d/dx((x - y)**2) = d/dx(x**2 -2xy + y**2) =
                     = d/dx(x**2) + d/dx(-2xy) + d/dx(y**2)) =
                     = 2x +(-2y) + 0 =
                     = 2x - 2y =
                     = 2(x - y),

    where x = predicted_output, y = target_output ❏
    */
    return 2 * (predicted_output - target_output);
}

/**
 Train the model on the given samlpes.
*/
void FullyConnectedNeuralNetwork::train(
    const std::vector<Tensor1D*>& training_samples,
    const std::vector<Tensor1D*>& training_labels,
    const float& learning_rate,
    const uint& n_epochs,
    const bool& verbose
) {
    assert(training_samples.size() == training_labels.size());

    uint n_samples = training_samples.size();

    std::cout << "training started ✓" << std::endl;

    // for each epoch:
    for (uint epoch_indx = 0; epoch_indx < n_epochs; ++epoch_indx) {
        std::cout << "epoch " << (epoch_indx + 1) << "/" << n_epochs
            << std::endl;

        // for each training sample:
        for (uint sample_indx = 0; sample_indx < n_samples; ++sample_indx) {
            // forward propagation of the sample through the network, computing
            // layers' activations sequentially:
            this->forwardPropagation(training_samples[sample_indx]);

            if (verbose) {
                std::cout << "\t" << "sample features: "
                    << *training_samples[sample_indx] << std::endl;
                std::cout << "\t" << "expected output: "
                    << *training_labels[sample_indx] << std::endl;
                std::cout << "\t" << "vs" << std::endl;
                std::cout << "\t" << "actual output: "
                    << *(this->activations.back()) << std::endl;
                std::cout << "\t" << "loss value: "
                    << squaredErrorLoss(this->activations.back(),
                                        training_labels[sample_indx])
                    << std::endl;
            }

            // backward propagation of the loss gradients through the network
            // with respect to the different weights eventually updating them
            // accordingly:
            this->backPropagation(training_labels[sample_indx], learning_rate);

            if (verbose) {
                std::cout << "\t" << "weights updated accordingly ✓"
                    << std::endl;
                // re-propagating the sample through the network to evaluate
                // the new loss value, after the weight update:
                this->forwardPropagation(training_samples[sample_indx]);
                std::cout << "\t" << "updated loss value: "
                    << squaredErrorLoss(this->activations.back(),
                                        training_labels[sample_indx])
                    << std::endl;
                std::cout << "__________________________" << std::endl;
            }
        }
    }

    std::cout << "model successfully trained ✓" << std::endl;
}
