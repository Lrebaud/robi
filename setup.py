from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name='robi',
    version='0.1.0',
    description='ROBI: Robust and Optimized Biomarker Identifier',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Lrebaud/robi',
    author='Louis Rebaud',
    author_email='louis.rebaud@gmail.com',
    license="Apache License 2.0",
    packages=['robi'],
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
        'scikit-learn',
        'lifelines',
        'tqdm',
        'statmodels',
        'ray',
        'multipy'
    ],
    tests_require=['torch', 'pytest'],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)