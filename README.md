<h1 align="center">
    <p>A C++ Implementation of a Fully-Connected Neural Network</p>
</h1>

A C++ implementation of a [feed-forward, fully-connected, multi-layer neural network](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/model.hpp#L1) for single-output regression problems, including both [forward](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/model.cpp#L341) and [backward](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/model.cpp#L148) propagation, supporting [settable hyperparameters](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/main.cpp#L13) such as activation functions' kind (Leaky ReLU, ReLU or Sigmoid) and number of layers and number of neurons each, [trained](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/model.cpp#L432) via Stochastic Gradient Descent (mini-batches of 1 sample) and [evaluated](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/model.cpp#L279) in terms of Mean Squared Error respectively on [training](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/main.cpp#L30) and [validation](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/main.cpp#L35) partitions of a [self-made non-linear regression dataset](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/create_dataset.py#L1): a 2D projection of a [Swiss Roll dataset by Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html), whose samples are shown as color-coded by output value in the 2D feature space in the following picture.

<p align="center">
    <img src="https://github.com/MattiaSarti/toy-neural-network-in-cpp/blob/main/readme_pictures/whole_dataset.png" alt="...loading..."  width="500"/>
</p>

Results on the validation set follow, presented visually (please, mind the colorbar), confirming the capability of the model to learn non-linear representations and the correctness of the implementation.
<br><br>

<p align="center">
    <img src="https://github.com/MattiaSarti/toy-neural-network-in-cpp/blob/main/readme_pictures/predictions_before_and_after_training.png" alt="...loading..."  width="1000"/>
</p>


## My Attitude

After studying C++ syntax and reviewing neural networks' theory (mostly [gradient computation](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/model.cpp#L187) for backpropagation), I designed the [model class](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/model.hpp#L1), made it computationally efficient, documented it (well-commented code) and added an example of [training](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/main.cpp#L59) and [evaluation](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/main.cpp#L66) on [such a self-made dataset](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/readme_pictures/separate_sets.png) (Python was used only for the [dataset creation](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/create_dataset.py#L1) and for [displaying plots](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/plot_validation_predictions.py#L1) because of its more suitable libraries, that came in handy), as an exercise to learn about C++ (and neural networks).


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

1. **Create the Dataset**

    First, run ```python create_dataset.py``` from the root folder of this project to create and save the above-mentioned dataset; these plots describing the created datasets will be displayed (and saved) as well:

    <p align="center">
        <img src="https://github.com/MattiaSarti/toy-neural-network-in-cpp/blob/main/readme_pictures/whole_dataset.png" alt="...loading..."  width="500"/>
    </p>

    <p align="center">
        <img src="https://github.com/MattiaSarti/toy-neural-network-in-cpp/blob/main/readme_pictures/separate_sets.png" alt="...loading..."  width="1200"/>
    </p>

2. **Create, Train and Validate the Model**

    TODO(me):  describe model, hyperparameters, training and validation, evaluation, MSE both as training loss and metric (which is possible as it is differentiable and represent a good metric), describe that next launch script that does everything, describe the hyperparameters that have been set

    I did some manual hyperparameter tuning beforehand to know how many layers and neurons in each one could yield good results on a similar dataset exploiting [TensorFlow Playground](https://playground.tensorflow.org/), and [these final settings that I tried](https://playground.tensorflow.org/#activation=relu&batchSize=1&dataset=spiral&regDataset=reg-gauss&learningRate=0.01&regularizationRate=0&noise=0&networkShape=8,8,6,4&seed=0.75558&showTestData=false&discretize=false&percTrainData=70&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false) convinced me to input [such architecture hyperparameters (number of layers and neurons each)](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/main.cpp##L15) to the script execution:

    ```
    g++ -c -I <your_eigen_library_path> -o activation_functions.o activation_functions.cpp
    g++ -c -I <your_eigen_library_path> -o load_dataset.o load_dataset.cpp
    g++ -c -I <your_eigen_library_path> -o model.o model.cpp
    g++ -c -I <your_eigen_library_path> -o main.o main.cpp
    g++ -o training_and_validation_executable activation_functions.o load_dataset.o model.o main.o
    training_and_validation_executable
    ```

    Logs illustrating the executed steps (and the validation metrics) will be printed to your standard output:

    ```
    training set loaded ✓
    validation set loaded ✓
    - - - - - - - - - - - -
    model initialized ✓
    - - - - - - - - - - - -
    MSE on the validation set: 88.5111
    - - - - - - - - - - - -
    training started ✓
    epoch 100/500
    epoch 200/500
    epoch 300/500
    epoch 400/500
    epoch 500/500
    model successfully trained ✓
    - - - - - - - - - - - -
    MSE on the validation set: 0.766667
    - - - - - - - - - - - -
    ```

    *Interestingly, I was not able to reach such a good MSE with ReLU and Sigmoid activation functions but only with [Leaky ReLU](https://github.com/MattiaSarti/neural-network-in-cpp/blob/main/main.cpp#L14), which 1) does not suffer from the vanishing/exploding gradient problems while Sigmoid does (same advantage as ReLU) and 2) prevents gradients from "dying" when inputs are negative contrarily to ReLU, as always reported in literature.*

3. **Evaluate Results Visually**

    Finally, plot the predictions on the validation set before and after training, to check what the model has learned, by running ```python plot_validation_predictions.py``` from the root folder:

    <p align="center">
        <img src="https://github.com/MattiaSarti/toy-neural-network-in-cpp/blob/main/readme_pictures/predictions_before_and_after_training.png" alt="...loading..."  width="1200"/>
    </p>

    This capability of learning non-linear patterns is quite charming, isn't it?

#### A Note on Reproducibility
Results are perfectly reproducible as all sources of randomness (dataset creation and splitting, weight initialization) have seen their seeds fixed.\
Moreover, the code is OS-independent and can be run anywhere with the above-mentioned requirements.


## Code Style

- **C++**
    - clang-tidy 12.0.0-6923b0a7 (LLVM) compliant ✔
    - cpplint 1.5.5 compliant ✔
- **Python**
    - flake8 3.8.4 compliant ✔
    - pylint 2.5.3 compliant ✔
