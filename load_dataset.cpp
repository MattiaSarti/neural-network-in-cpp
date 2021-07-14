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

    // counting the number of columns (and features):

    getline(file, line, '\n');

    stringstream stream(line);

    uint n_features = 0;
    while (getline(stream, cell, ',')) {
        cout << float(stof(&cell[0])) << endl;
        ++n_features;
    }
    // since the last column represents the class label, the actual number of
    // features equal the number of columns encountered minus one:
    n_features -= 1;
    cout << n_features << endl;

    if (file.is_open()) {

        // reading the following rows, the actual ones with data, each one
        // containing a sample's feature values and its label:

        while (getline(file, line, '\n')) {

            stringstream stream(line);

            samples.push_back(new Tensor1D(n_features));
            labels.push_back(new Tensor1D(1));

            uint feature_indx = 0;
            while (getline(stream, cell, ',')) {

                // if the current cell is not the last one of the row, it
                // contains the corresponding feature's value:
                if (feature_indx != n_features - 1) {

                    // adding the feature value to the sample:
                    samples.back()->coeffRef(feature_indx) = float(stof(&cell[0]));

                // if the current cell is the last one of the row, it contains
                // the class label:
                } else {

                    // associating the class label to the sample:
                    labels.back()->coeffRef(1) = float(stof(&cell[0]));

                }

                ++feature_indx;
            }

        }

    }

}

int main () {
    vector<Tensor1D*> samples;
    vector<Tensor1D*> labels;
    loadDataset("temp.csv", samples, labels);
    for (uint i = 0; i < samples.size(); ++i) {
        cout << *samples[i] << " - ";
    }
    cout << endl;
    for (uint i = 0; i < labels.size(); ++i) {
        cout << *labels[i] << " - ";
    }

    return 0;
}
