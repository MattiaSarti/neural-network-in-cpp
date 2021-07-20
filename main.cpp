/*
 Script loading the training and validation sets from the file system,
 training the model on the training set and evaluating it before and after
 training on the validation set, saving results to the file system.
*/


#include "common.hpp"  // "model.hpp"  // FIXME


const float LEARNING_RATE = 0.001;
const uint N_EPOCHS = 10000;


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

    std::vector<uint> n_neurons_in_layers = {2, 4, 3};
    // asserting that the number of neurons of first layer equals the number
    // of features that samples in the dataset have:
    assert(n_neurons_in_layers.front() == validation_samples.front()->size());
    FullyConnectedNeuralNetwork model(n_neurons_in_layers);

    std::cout << "- - - - - - - - - - - -" << std::endl;

    // evaluating the model on the validation set before training:
    std::string results_path = "validation_set_predictions_before_training.csv";
    model.evaluate(validation_samples, validation_labels, results_path, false);

    std::cout << "- - - - - - - - - - - -" << std::endl;

    // training the model on the training set:
    model.train(training_samples, training_labels, LEARNING_RATE, N_EPOCHS,
        false);

    std::cout << "- - - - - - - - - - - -" << std::endl;

    // evaluating the model on the validation set after training:
    results_path = "validation_set_predictions_after_training.csv";
    model.evaluate(validation_samples, validation_labels, results_path, false);

    std::cout << "- - - - - - - - - - - -" << std::endl;

    return 0;
}
