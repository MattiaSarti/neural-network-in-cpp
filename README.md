# A C++ Implementation of a Simple Fully-Connected Neural Network

A C++ implementation of a feed-forward, fully-connected, multi-layer neural network for single-output regression problems, trained via Stochastic Gradient Descent (mini-batches of 1 sample) and evaluated respectively on training and validation partitions of a self-made regression dataset: a 2D projection of a Swiss Roll dataset by Scikit-Learn, whose samples are shown as color-coded by output value in the 2D feature space in the following picture.

<img src="https://github.com/MattiaSarti/toy-neural-network-in-cpp/blob/main/readme_pictures/bigger_dataset.png" alt="...loading..." width="200"/>


## How to Reproduce Results:

```
python create_dataset.py
```

<img src="https://github.com/MattiaSarti/toy-neural-network-in-cpp/blob/main/readme_pictures/whole_dataset.png" alt="...loading..." width="200"/>


```
g++
```

- MAE as metric

```
python plot_validation_predictions.py
```

<img src="https://github.com/MattiaSarti/toy-neural-network-in-cpp/blob/main/readme_pictures/predictions_before_and_after_training.png" alt="...loading..." width="200"/>


## Requirements

- C++
    - gcc Version: 6.3.0 (MinGW.org GCC-6.3.0-1)
    - Libraries (& Their Versions):
        - eigen==3.3.9
- Python
    - Version: 3.8.0
    - Libraries (& Their Versions):
        - numpy==1.20.2
        - matplotlib==3.3.2
        - scikit_learn==0.24.2


## My Attitude

I went through [an already-existing implementation](https://www.geeksforgeeks.org/ml-neural-network-implementation-in-c-from-scratch/) first and then I 1) modified it to make it more efficient and 2) extended it by adding training and evaluation on a self-made dataset, as an exercise to learn about C++ (and neural networks).
