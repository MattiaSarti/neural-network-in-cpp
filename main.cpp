/*
 Bla bla bla.
*/


#include "common.hpp"//"model.hpp"  // FIXME


int main ()
{

    // loading training and validation sets:
    vector<Tensor1D*> training_samples;
    vector<Tensor1D*> training_labels;
    loadDataset("training_set.csv", training_samples, training_labels);
    cout << "training set loaded ✓" << endl;

    vector<Tensor1D*> validation_samples;
    vector<Tensor1D*> validation_labels;
    loadDataset("validation_set.csv", validation_samples, validation_labels);
    cout << "validation set loaded ✓" << endl;

    /*
    for (uint i = 0; i < samples.size(); ++i) {
        cout << *samples[i] << " - ";
    }
    cout << endl;
    for (uint i = 0; i < labels.size(); ++i) {
        cout << *labels[i] << " - ";
    }
    */

    cout << "- - - - - - - - - - - -" << endl;

    vector<uint> n_neurons_in_layers = {2, 4, 3};
    // asserting that the number of neurons of first layer equals the number
    // of features that samples in the dataset have:
    // TODO: assert(n_neurons_in_layers.front() == validation_samples.front().size());
    FullyConnectedNeuralNetwork model(n_neurons_in_layers);

    cout << "- - - - - - - - - - - -" << endl;

    // evaluating the model on the validation set before training:
    string results_path = "validation_set_predictions_before_training.csv";
    model.evaluate(validation_samples, validation_labels, results_path);

    cout << "- - - - - - - - - - - -" << endl;

    // training the model on the training set:
    float learning_rate = 0.001;
    uint n_epochs = 10000;
    model.train(training_samples, training_labels, learning_rate, n_epochs);

    cout << "- - - - - - - - - - - -" << endl;

    // evaluating the model on the validation set after training:
    results_path = "validation_set_predictions_after_training.csv";
    model.evaluate(validation_samples, validation_labels, results_path);

    cout << "- - - - - - - - - - - -" << endl;

    return 0;
}
