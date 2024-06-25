# Credit Card Consumption Prediction

This project predicts credit card consumption using various machine learning algorithms.

## Directory Structure

- `ccp/src`: Source code for data preprocessing, feature engineering, model training, and evaluation.
- `ccp/test`: Unit tests for the source code.
- `ccp/data`: Directory for storing datasets.
- `.gitignore`: Specifies files and directories to be ignored by git.
- `README.md`: Project description and instructions.
- `requirements.txt`: List of dependencies required to run the project.
- `setup.py`: Configuration for packaging the project.
- `main.py`: Main script to run the data processing and model training pipeline.

## Installation

```bash
pip install -r requirements.txt
python setup.py install
```

## Usage
To run the main script with a configurable prediction window:
- `python main.py --window week`
- `Options for --window: day, week, month`

## Tests
- `pytest ccp/test`
