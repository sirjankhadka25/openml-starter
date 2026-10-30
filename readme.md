
# OpenML Starter

A beginner-friendly machine learning project demonstrating a reproducible classification pipeline using Python and scikit-learn.

## Features

- Wine classification using Logistic Regression
- Feature standardization with StandardScaler
- Reproducible train/test splitting
- Model evaluation using accuracy and confusion matrix
- Model saving and loading using joblib

## Tech Stack

- Python
- scikit-learn
- NumPy
- joblib

## Installation

Clone the repository and open its directory.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train the Model

```bash
python train.py
```

This trains and evaluates the classification model, then saves the pipeline to `models/wine_model.joblib`.

## Make Predictions

After training:

```bash
python predict.py
```

This loads the saved pipeline and predicts the class of a sample wine.

## Dataset

Uses the Wine classification dataset provided by scikit-learn.

- 178 samples
- 13 numerical features
- 3 target classes

## Current Results

Initial test accuracy: approximately 97.22% (35/36 correct predictions).

This result is based on a single train/test split and is not a guarantee of future performance.

## Future Improvements

- Automated testing
- Cross-validation
- Configuration management
- Model comparison
- Command-line interface
- API deployment

## License

License to be selected before public release.
