from setuptools import setup, find_packages

setup(
    name='ccp',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn'
    ],
)
