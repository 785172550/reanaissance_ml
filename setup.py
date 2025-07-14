# setup.py
from setuptools import setup, find_packages


setup(
    name='hw_fin_utils',
    version='0.1.0',
    description='A sample Python library using uv lock',
    packages=find_packages(),
    author='Hang Wang',
    author_email='hangw9412@gmail.com',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'pandas_ta',
        'scikit-learn',
        'yfinance'
    ],
    python_requires='>=3.8',
)
