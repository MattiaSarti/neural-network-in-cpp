"""
This script creates a dataset for a single-output regression problem on a 2D
input feature space.
"""


from os import getcwd
from os.path import join as os_join

from matplotlib.pyplot import (
    colorbar, figure, savefig, scatter, show, title, xlabel, ylabel
)
from numpy import savetxt, stack
from sklearn.datasets import make_swiss_roll


GAUSSIAN_NOISE_STD = 0.2
N_SAMPLES = 600
SEED = 0

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

    # plotting the dataset samples in the 2D space color-coded by label:
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

    # saving the dataset to a CSV file where each row - but the first one,
    # which is used only to count the number of columns - represents a sample
    # whose respective features and its label are orderly reported in
    # different columns:
    dataset_path = os_join(
        PROJECT_DIR,
        "dataset.csv"
    )
    dataset = stack(
        (
            dataset_samples[:, 0],
            dataset_samples[:, 2],
            dataset_labels
        ),
        axis=-1
    )
    savetxt(fname=dataset_path, X=dataset, delimiter=',')
