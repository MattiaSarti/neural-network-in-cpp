# A C++ Implementation of a Simple Fully-Connected Neural Network

- Feed-Forward Fully-Connected Multi-Layer
- Trained
- SGD (mini-batches of a 1 sample)
- Regression
- Dataset (a 2D projection of a Swiss Roll dataset by Scikit-Learn)
- MAE on validation set for evaluation

![...loading...](https://github.com/MattiaSarti/toy-neural-network-in-cpp/blob/main/readme_pictures/bigger_dataset.png)

![...loading...](https://github.com/MattiaSarti/toy-neural-network-in-cpp/blob/main/readme_pictures/whole_dataset.png)

![...loading...](https://github.com/MattiaSarti/toy-neural-network-in-cpp/blob/main/readme_pictures/separate_sets.png)

![...loading...](https://github.com/MattiaSarti/toy-neural-network-in-cpp/blob/main/readme_pictures/predictions_before_and_after_training.png)


## Requirements

- C++
    - gcc Version: 6.3.0 (MinGW.org GCC-6.3.0-1)\
    - Libraries (& Their Versions):
        - eigen==3.3.9

- Python
    - Version: 3.8.0\
    - Libraries (& Their Versions):
        - numpy==1.20.2
        - matplotlib==3.3.2
        - scikit_learn==0.24.2


## My Attitude

I went through [an already-existing implementation](https://www.geeksforgeeks.org/ml-neural-network-implementation-in-c-from-scratch/) first and then I 1) modified it to make it more efficient and 2) extended it by adding training and evaluation on a self-made dataset, as an exercise to learn about C++ (and neural networks).
