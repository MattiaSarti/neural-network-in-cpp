/*
 Bla bla bla.
*/


#include <common.hpp>


using namespace std;


/*
 Load the samples and labels constituting the dataset from a CSV file where
 each row - but the first one, which is used only to count the number of
 columns - representa a sample: each cell contains a feature value but the
 last one, which represents the class label.
*/
void loadDataset (string filename, vector<Tensor1D*>& samples, vector<Tensor1D*>& labels) {

    string line, cell;

    samples.clear();
    labels.clear();
    ifstream file(filename);

    // counting the number of columns (and features):

    getline(file, line, '\n');
    stringstream stream(line);
    uint n_features = 0;
    while (getline(stream, cell, ',')) {
        ++n_features;
    }
    // since the last column represents the class label, the actual number of
    // features equal the number of columns encountered minus one:
    n_features -= 1;

    if (file.is_open()) {

        // reading the following rows - the actual ones with data - each one
        // containing a sample's feature values (all but the last cell) and
        // its label (the last cell):

        while (getline(file, line, '\n')) {

            stringstream stream(line);

            samples.push_back(new Tensor1D(n_features));
            labels.push_back(new Tensor1D(1));

            float cell_value;
            uint feature_indx = 0;
            while (getline(stream, cell, ',')) {

                cell_value = float(stof(&cell[0]));

                // if the current cell is not the last one of the row, it
                // contains the corresponding feature's value:
                if (feature_indx != n_features) {

                    // adding the feature value to the sample:
                    samples.back()->coeffRef(feature_indx) = cell_value;

                // if the current cell is the last one of the row, it contains
                // the class label:
                } else {

                    // associating the class label to the sample:
                    labels.back()->coeffRef(0) = cell_value;

                }

                ++feature_indx;
            }

        }

    }

}
