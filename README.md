# MNIST-Digit-Classification-KNN
<img width="741" alt="Screenshot 2023-10-12 at 8 34 32 AM" src="https://github.com/ImrulNYC/MNIST-Digit-Classification-KNN/assets/147569091/5a39e78e-cea5-4359-a25d-c8297e3ce9f1"><img width="741" alt="Screenshot 2023-10-12 at 8 34 46 AM" src="https://github.com/ImrulNYC/MNIST-Digit-Classification-KNN/assets/147569091/09a027ef-139c-4ca1-8040-d00ac0b78ef9"><img width="741" alt="Screenshot 2023-10-12 at 8 34 53 AM" src="https://github.com/ImrulNYC/MNIST-Digit-Classification-KNN/assets/147569091/76cadf86-49a2-4dd0-bb31-6bedcd9a6d1d">

Run using https://colab.research.google.com
#or
Make sure you have the following Python libraries installed:

numpy
pandas
matplotlib
pandas-datareader
scikit-learn
tensorflow
install these libraries using pip:

pip install numpy pandas matplotlib pandas-datareader scikit-learn tensorflow


Certainly, here's a README template for your project:

# MNIST Digit Classification with k-Nearest Neighbors (k-NN)

## Overview

This project focuses on digit classification using the MNIST dataset, a popular dataset for digit recognition. We implement a k-Nearest Neighbors (k-NN) classifier to classify digits as either "5" or "not 5" (binary classification). 
The project involves data preprocessing, model training, and hyperparameter tuning to achieve the best possible F1 score.


## Data

We use the MNIST dataset obtained from the `fetch_openml` function in scikit-learn. The dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9). We preprocess the dataset to perform binary classification by labeling digits as either "5" or "not 5."

## Data Split

The dataset is split into a training set (80%) and a test set (20%) using `train_test_split` from scikit-learn. This division ensures that the model is trained and evaluated on different data.

## Model

Used a k-Nearest Neighbors (k-NN) classifier for this binary classification task. 
The model's hyperparameters are optimized using Grid Search with cross-validation.

### Hyperparameter Tuning

tuned the following hyperparameters:
- `n_neighbors`: Number of neighbors to consider.  choose the best value through hyperparameter tuning.
- `weights`: Weight function used in prediction.  used the 'uniform' option for this project.

## Results

After tuning the model's hyperparameters, we evaluate its performance on the test set using the F1 score.
The F1 score is a measure of a model's accuracy in binary classification. 
The project reports the model's F1 score on the test set.

