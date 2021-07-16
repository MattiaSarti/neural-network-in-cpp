"""
This script plots the predictions over the samples of the validation set
before and after training.
"""


from os import getcwd
from os.path import join as os_join

from matplotlib.pyplot import figure, savefig, show, subplot, title
from numpy import array, ndarray

from create_dataset import create_dataset, plot, split_dataset


PROJECT_DIR = getcwd()
PRE_TRAINING_PREDICTIONS_PATH = os_join(
    PROJECT_DIR,
    "validation_set_predictions_before_training.csv"
)
POST_TRAINING_PREDICTIONS_PATH = os_join(
    PROJECT_DIR,
    "validation_set_predictions_after_training.csv"
)
PLOT_PICTURE_PATH = os_join(
    PROJECT_DIR,
    "readme_pictures",
    "predictions_before_and_after_training.png"
)


def load_predictions(file_path: str) -> ndarray:
    """
    Load the validation predictions from the specified file.
    """
    predictions = []
    with open(file_path, 'r') as file:
        for line_text in file:
            predictions.append(float(line_text))
    predictions = array(predictions)
    return predictions


if __name__ == '__main__':

    # re-creating the same validation dataset employed previously:
    dataset = create_dataset()
    _, validation_dataset = split_dataset(
        whole_dataset=dataset
    )

    # loading the (ordered) predictions over the validation set, both the ones
    # made before and the ones made after training:
    pre_training_predictions = load_predictions(
        file_path=PRE_TRAINING_PREDICTIONS_PATH
    )
    post_training_predictions = load_predictions(
        file_path=POST_TRAINING_PREDICTIONS_PATH
    )
    
    fig = figure(figsize=(12, 4))
    fig.suptitle("Predictions On The Validation Set")

    # replacing the target labels with the predicitons made by the model
    # before training:
    validation_dataset[:, 2] = pre_training_predictions

    # plotting the predictions made before training:
    subplot(1, 2, 1)
    plot(data=validation_dataset, title_text="Before Training")

    # replacing the target labels with the predicitons made by the model
    # after training:
    validation_dataset[:, 2] = post_training_predictions

    # plotting the predictions made after training:
    subplot(1, 2, 2)
    plot(data=validation_dataset, title_text="After Training")

    savefig(PLOT_PICTURE_PATH)
    show()
