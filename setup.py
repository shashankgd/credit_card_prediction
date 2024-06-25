from setuptools import setup, find_packages

setup(
    name='credit_card_prediction',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'xgboost',
        'lightgbm',
        'pytest',
        'matplotlib',
        'seaborn'
    ],
    entry_points={
        'console_scripts': [
            'run-credit-card-prediction=main:main',
        ],
    },
)
