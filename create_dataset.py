"""
This script creates a dataset for a single-output regression problem on a 2D
input feature space.
"""


from os import getcwd
from os.path import join as os_join

from matplotlib.pyplot import (
    colorbar, figure, savefig, scatter, show, title, xlabel, ylabel
)
from numpy import insert, savetxt, stack
from sklearn.datasets import make_swiss_roll


GAUSSIAN_NOISE_STD = 0.2
N_SAMPLES = 10
SEED = 0
VALIDATION_AMOUNT = 0.3

PLOT = True
SAVE_PLOT = True
SAVE_DATA = True

PROJECT_DIR = getcwd()


# creating the dataset:
dataset_samples, dataset_labels = make_swiss_roll(
    n_samples=N_SAMPLES,
    noise=GAUSSIAN_NOISE_STD,
    random_state=SEED
)

if PLOT or SAVE_PLOT:

    # plotting the dataset samples in the 2D feature space, color-coded by
    # label:
    figure()
    scatter(
        x=dataset_samples[:, 0],
        y=dataset_samples[:, 2],
        c=dataset_labels
    )
    xlabel("Feature 1")
    ylabel("Feature 2")
    title("Output Value as Color")
    colorbar()

    if SAVE_PLOT:

        # saving the plot picture:
        picture_path = os_join(
            PROJECT_DIR,
            "readme_pictures",
            "dataset_plot.png"
        )
        savefig(picture_path)

    if PLOT:

        # displaying the plot:
        show()

if SAVE_DATA:

    # splitting the dataset into training and validation sets:
    dataset = stack(
        (
            dataset_samples[:, 0],
            dataset_samples[:, 2],
            dataset_labels
        ),
        axis=-1
    )
    split_sign = int(N_SAMPLES * (1 - VALIDATION_AMOUNT))
    training_dataset = dataset[:split_sign,]
    validation_dataset = dataset[split_sign:,]

    # saving each dataset to a CSV file where each row - but the first one,
    # which is used only to count the number of columns - represents a sample
    # whose respective features and its label are orderly reported in
    # different columns:
    training_dataset_path = os_join(
        PROJECT_DIR,
        "training_set.csv"
    )
    validation_dataset_path = os_join(
        PROJECT_DIR,
        "validation_set.csv"
    )
    # adding a fictitious initial sample with no data to each file just to
    # count features:
    training_dataset = insert(
        arr=training_dataset,
        obj=0,
        values=[None, None, None],
        axis=0
    )
    validation_dataset = insert(
        arr=validation_dataset,
        obj=0,
        values=[None, None, None],
        axis=0
    )
    savetxt(fname=training_dataset_path, X=training_dataset, delimiter=',')
    savetxt(fname=validation_dataset_path, X=validation_dataset, delimiter=',')
