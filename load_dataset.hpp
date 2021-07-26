/*
 Utilities for loading the datasets from the file system.
*/


#ifndef LOAD_DATASET_HPP
#define LOAD_DATASET_HPP


#include "common.hpp"


/**
 Load the samples and labels constituting the dataset from a CSV file where
 each row - but the first one, which is used only to count the number of
 columns - representa a sample: each cell contains a feature value but the
 last one, which represents the class label.
*/
void loadDataset(
    const std::string& filename,
    std::vector<Tensor1D*>* samples,
    std::vector<Tensor1D*>* labels);


#endif
