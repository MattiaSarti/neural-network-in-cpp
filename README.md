<h1 align="center">
    <p>A C++ Implementation of a Fully-Connected Neural Network</p>
</h1>

A C++ implementation of a feed-forward, fully-connected, multi-layer neural network for single-output regression problems, trained via Stochastic Gradient Descent (mini-batches of 1 sample) and evaluated respectively on training and validation partitions of a self-made non-linear regression dataset: a 2D projection of a Swiss Roll dataset by Scikit-Learn, whose samples are shown as color-coded by output value in the 2D feature space in the following picture.

<p align="center">
    <img src="https://github.com/MattiaSarti/toy-neural-network-in-cpp/blob/main/readme_pictures/bigger_dataset.png" alt="...loading..."  width="500"/>
</p>


## My Attitude

What I have implemented... After reviewing backpropagation theory, ... designed it, made it computationally efficient, documented it (well-commented code) and added training and evaluation on such a self-made dataset, as an exercise to learn about C++ (and neural networks).


## How to Reproduce Results

### Requirements:

- **C++**
    - gcc Version: 6.3.0 (MinGW.org GCC-6.3.0-1)
    - Libraries (& Their Versions):
        - eigen==3.3.9
- **Python - Only for Creating Datasets and Plots**
    - Version: 3.8.0
    - Libraries (& Their Versions):
        - numpy==1.20.2
        - matplotlib==3.3.2
        - scikit_learn==0.24.2

### Steps:

```python create_dataset.py```

<p align="center">
    <img src="https://github.com/MattiaSarti/toy-neural-network-in-cpp/blob/main/readme_pictures/whole_dataset.png" alt="...loading..."  width="500"/>
</p>

<p align="center">
    <img src="https://github.com/MattiaSarti/toy-neural-network-in-cpp/blob/main/readme_pictures/separate_sets.png" alt="...loading..."  width="900"/>
</p>



I did some manual hyperparameter tuning beforehand to know how many layers and neurons in each one could yield good results on a similar dataset exploiting [TensorFlow Playground](https://playground.tensorflow.org/), and [these final settings that I tried](https://playground.tensorflow.org/#activation=relu&batchSize=1&dataset=spiral&regDataset=reg-gauss&learningRate=0.01&regularizationRate=0&noise=0&networkShape=8,8,6,4&seed=0.75558&showTestData=false&discretize=false&percTrainData=70&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false) convinced me to input such architecture hyperparameters (number of layers and neurons each) to the script execution:


```
g++ ...
```
- MSE both as loss and metric

```python plot_validation_predictions.py```

<p align="center">
    <img src="https://github.com/MattiaSarti/toy-neural-network-in-cpp/blob/main/readme_pictures/predictions_before_and_after_training.png" alt="...loading..."  width="900"/>
</p>


## Code Style

- **C++**
    - clang-tidy 12.0.0-6923b0a7 (LLVM) compliant ✔
    - cpplint 1.5.5 compliant ✔
- **Python**
    - flake8 3.8.4 compliant ✔
    - pylint 2.5.3 compliant ✔
