/*
 Bla bla bla.
*/


// TODO: if not def include


int main () {

    vector<Tensor1D*> samples;
    vector<Tensor1D*> labels;

    loadDataset("temp.csv", samples, labels);

    /*
    for (uint i = 0; i < samples.size(); ++i) {
        cout << *samples[i] << " - ";
    }
    cout << endl;
    for (uint i = 0; i < labels.size(); ++i) {
        cout << *labels[i] << " - ";
    }
    */

    return 0;

}
