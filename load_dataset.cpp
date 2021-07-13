/*
 Bla bla bla.
*/


#include <common.hpp>


using namespace std;


void loadDataset (string filename, vector<Tensor1D*>& samples, vector<Tensor1D*>& labels) {

    samples.clear();
    labels.clear();

    string line, cell;
    vector<float> first_sample;

    ifstream file(filename);

    getline(file, line, '\n');
    stringstream stream(line);
    while (getline(stream, cell, ', ')) {
        first_sample.push_back(float(stof(&cell[0])));
    }

    uint n_features = first_sample.size();

    samples.push_back(new Tensor1D(n_features));
    labels.push_back(Tensor1D(1));

    for (uint feature_indx = 0; feature_indx < n_features; ++feature_indx) {

        if (feature_indx != n_features - 1) {
            samples.back()->coeffRef(feature_indx) = first_sample[feature_indx];
        } else {
            labels.back()->coeffRef(1) = first_sample[feature_indx];
        }

    }

    // reading the following rows, each one containing a sample and its label:

    if (file.is_open()) {

        while (getline(file, line, '\n')) {

            stringstream stream(line);
            samples.push_back(new Tensor1D(n_features));

            uint feature_indx = 0;
            while (getline(stream, cell, ', ')) {
                samples.back()->coeffRef(feature_indx) = float(stof(&cell[0]));
                ++feature_indx;
            }

        }

    }

}
