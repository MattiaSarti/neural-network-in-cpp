"""
This script creates a dataset for a single-output regression problem on a 2D
input feature space.
"""


from os import getcwd
from os.path import join as os_join
from typing import Tuple

from matplotlib.pyplot import (
    colorbar, figure, savefig, scatter, show, subplot, title, xlabel, ylabel
)
from numpy import insert, ndarray, savetxt, stack
from numpy.random import seed, shuffle
from sklearn.datasets import make_swiss_roll


GAUSSIAN_NOISE_STD = 1
N_SAMPLES = 600  # 80
SEED = 0
VALIDATION_AMOUNT = 0.3

PLOT = True
SAVE_DATA = True
SAVE_PLOTS = True

PROJECT_DIR = getcwd()


def create_dataset() -> ndarray:
    """
    Create the dataset samples and labels in a reproducible way.
    """
    # employing a Swiss Roll dataset by Scikit-Learn, with 3 input features
    # and a single numerical output:
    samples, labels = make_swiss_roll(
        n_samples=N_SAMPLES,
        noise=GAUSSIAN_NOISE_STD,
        random_state=SEED
    )
    # taking a 2D projection of the Swiss Roll dataset, stacking
    # samples and labels in the same array:
    stacked_samples_and_labels = stack(
        (
            samples[:, 0],
            samples[:, 2],
            labels
        ),
        axis=-1
    )
    return stacked_samples_and_labels


def plot(data: ndarray, title_text: str):
    """
    Create a custom scatter plot with the desired data and title while
    following the same template for axes' labels and colorbar.
    """
    scatter(
        x=data[:, 0],
        y=data[:, 1],
        c=data[:, 2]
    )
    dress_plot()
    title(title_text)


def dress_plot() -> None:
    """
    Add axes' labels and colorbar to the current plot.
    """
    xlabel("Feature 1")
    ylabel("Feature 2")
    colorbar()


def save_dataset(actual_dataset: ndarray, file_path: str) -> None:
    """
    Save the input dataset to a CSV file where each row - but the first one,
    which is used only to count the number of columns - represents a sample
    whose respective features and its label are orderly reported in different
    columns; a fictitious initial sample with no data is added to the file
    just to count features:
    """
    # adding a fictitious initial sample with no data to file just to allow to
    # count features more conveniently - creatind kind of an "initially
    # padded" dataset:
    padded_dataset = insert(
        arr=actual_dataset,
        obj=0,
        values=[None, None, None],
        axis=0
    )
    # saving the padded dataset to the file system:
    savetxt(
        fname=file_path,
        X=padded_dataset,
        delimiter=','
    )


def split_dataset(whole_dataset: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Split the dataset into training and validation sets, after randomly
    shuffling samples and labels (keeping correspondences).
    """
    seed(0)  # for a reproducible shuffling
    shuffle(whole_dataset)
    split_sign = int(N_SAMPLES * (1 - VALIDATION_AMOUNT))
    training_set = whole_dataset[:split_sign, ]
    validation_set = whole_dataset[split_sign:, ]
    return training_set, validation_set


if __name__ == '__main__':

    # creating the dataset:
    dataset = create_dataset()

    # splitting the dataset into training and validation sets, after randomly
    # shuffling samples and labels (keeping correspondences):
    training_dataset, validation_dataset = split_dataset(
        whole_dataset=dataset
    )

    if PLOT or SAVE_PLOTS:

        # plotting the whole dataset samples in the 2D feature space,
        # color-coded by label:
        figure()
        plot(data=dataset, title_text="Output Value as Color")

        if SAVE_PLOTS:

            # saving the plot picture:
            picture_path = os_join(
                PROJECT_DIR,
                "readme_pictures",
                "whole_dataset.png"
            )
            savefig(picture_path)

        if PLOT:

            # displaying the plot:
            show()

        # separately plotting the two separate sets' samples in the 2D feature
        # space, color-coded by label:
        figure(figsize=(12, 4))
        subplot(1, 2, 1)
        plot(data=training_dataset, title_text="Training Set")
        subplot(1, 2, 2)
        plot(data=validation_dataset, title_text="Validation Set")

        if SAVE_PLOTS:

            # saving the plot picture:
            picture_path = os_join(
                PROJECT_DIR,
                "readme_pictures",
                "separate_sets.png"
            )
            savefig(picture_path)

        if PLOT:

            # displaying the plot:
            show()

    if SAVE_DATA:

        # saving each dataset to a CSV file where each row - but the first
        # one, which is used only to count the number of columns - represents
        # a sample whose respective features and its label are orderly
        # reported in different columns; a fictitious initial sample
        # with no data is added to each file to count features:
        save_dataset(
            actual_dataset=training_dataset,
            file_path=os_join(
                PROJECT_DIR,
                "training_set.csv"
            )
        )
        save_dataset(
            actual_dataset=validation_dataset,
            file_path=os_join(
                PROJECT_DIR,
                "validation_set.csv"
            )
        )
