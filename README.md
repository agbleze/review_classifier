# review_classifier
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/agbleze/review_classifier/.github%2Fworkflows%2Fci-cd.yml)
![GitHub Tag](https://img.shields.io/github/v/tag/agbleze/review_classifier)
![GitHub Release](https://img.shields.io/github/v/release/agbleze/review_classifier)
![GitHub License](https://img.shields.io/github/license/agbleze/review_classifier)


## 📌 Table of Contents

- [Project Description](#project-description)
- [Dataset and Variables](#dataset-and-variables)
- [Model training pipeline](#model-training-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Credits](#credits)

---

# Project Description

This is a Natural Language Processing project for predicting whether 
a product will be recommended based on product review. A Deep Neural Network model design as a simple Convolutional Neural Network is presented for the task.

The value this project seeks to deliver is to enable businesses analyze their product reviews with respect to:

1. Determine after-sales and purchase intent of customers

2. To augment other sales data for prediction tasks

3. To improve customer satisfaction by automatically identifying and resolving issues

4. To augment and enrich other text datasets for advance analysis


## Dataset and Variables

Dataset: Product reviews provided by customers and clients after ordering products or patronizing services. The dataset is comprised of reviews on variety of products and services offered by different business entities and sectors.
                                     

1. Target variable: Recommendation status with binary values depicting whether or not a product was recommended

2. Predictor variable: Review of product which is text of variable length


## Model training pipeline

The model training procedure encompasses preprocessing of reviews, option to use pretrained embedding such as glove or initialize one and training CNN on the transformed data.

- Loss: CrosssEntropyLoss
- Optimizer: Adam


## Installation

1. Create and activate a virtual enviornment. This can be done on unbuntu based system as follows:

- I. Virtual environment called env (name it with your preference)

``` python -m venv env ```

- II. Activate the virtual environment

``` source env/bin/activate ```


2. Clone the repository as follows

```git clone https://github.com/agbleze/review_classifier.git ```

```bash
$ pip install .
```

## Usage

The package provides a high level entrypoint to train model from the terminal. Detailed demonstration of this including dataset format expected is provided in docs/example.ipynb


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`review_classifier` was created by Agbleze. It is licensed under the terms of the MIT license.

## Credits

`review_classifier` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
