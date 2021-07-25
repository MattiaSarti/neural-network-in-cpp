/*
 Script loading the training and validation sets from the file system,
 training the model on the training set and evaluating it before and after
 training on the validation set, saving results to the file system.
*/


#include "common.hpp"  // "model.hpp"  // FIXME


// architecture hyperparameters:
const auto ACTIVATION_FUNCTIONS_NAME = "ReLU";
const std::vector<uint> N_NEURONS_IN_LAYERS = {2, 8, 8, 6, 4};

// training hyperparameters:
const float LEARNING_RATE = 0.0001;
const uint N_EPOCHS = 10;

// display options:
const bool VERBOSE = false;


int main() {
    // loading training and validation sets:
    std::vector<Tensor1D*> training_samples;
    std::vector<Tensor1D*> training_labels;
    loadDataset("training_set.csv", &training_samples, &training_labels);
    std::cout << "training set loaded ✓" << std::endl;

    std::vector<Tensor1D*> validation_samples;
    std::vector<Tensor1D*> validation_labels;
    loadDataset("validation_set.csv", &validation_samples, &validation_labels);
    std::cout << "validation set loaded ✓" << std::endl;

    std::cout << "- - - - - - - - - - - -" << std::endl;

    // asserting that the number of neurons of first layer equals the number
    // of features that samples in the dataset have:
    assert(N_NEURONS_IN_LAYERS.front() == validation_samples.front()->size());
    FullyConnectedNeuralNetwork model(N_NEURONS_IN_LAYERS,
                                      ACTIVATION_FUNCTIONS_NAME);

    std::cout << "- - - - - - - - - - - -" << std::endl;

    // evaluating the model on the validation set before training:
    std::string results_path = "validation_set_predictions_before_training.csv";
    model.evaluate(validation_samples, validation_labels, results_path,
                   VERBOSE);

    std::cout << "- - - - - - - - - - - -" << std::endl;

    // training the model on the training set:
    model.train(training_samples, training_labels, LEARNING_RATE, N_EPOCHS,
        VERBOSE);

    std::cout << "- - - - - - - - - - - -" << std::endl;

    // evaluating the model on the validation set after training:
    results_path = "validation_set_predictions_after_training.csv";
    model.evaluate(validation_samples, validation_labels, results_path,
                   VERBOSE);

    std::cout << "- - - - - - - - - - - -" << std::endl;

    return 0;
}
