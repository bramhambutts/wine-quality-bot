# Wine Quality Bot

The purpose of this project is to use data on the classification of wines to determine the overall quality of the wine on a scale of 1-10. It should then be able to predict the quality of a wine input by a user with some degree of accuracy.

## The Data

The data for this project is from the UC Irvine Machine Learning Repository, first introduced by Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). This is available on the [UCI site](https://doi.org/10.24432/C56S3T) and from [their paper](https://www.semanticscholar.org/paper/Modeling-wine-preferences-by-data-mining-from-Cortez-Cerdeira/bf15a0ccc14ac1deb5cea570c870389c16be019c).

## Downloading the Project

This project relies on Python so a Python-3 distribution is required. For the commands in this project, the environment variable `python` has been used rather than `python3`, so substitutions may be required.

To download the project, navigate to the desired directory and use the following clone command:

```
git clone https://github.com/bramhambutts/wine-quality-bot.git
```

To set up the environment and prepare the project, use the following commands from within the `wine-quality-bot` directory:

```
python -m venv venv
source venv/bin/activate
export PYTHONPATH=$(pwd)
pip install -r requirements.txt
```

The project is now set up and possible to run and use.

## Using the classifier

To use the classifier to classify your wine, based on the parameters from the database, simply run `python src/simple_interface.py`. This will start a simple CLI for entering the data and seeing the predicted quality of the wine. It should also download and create the appropriate data as well as the model, allowing you to seamlessly use it without having to run more.