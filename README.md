[![PyPI version](https://badge.fury.io/py/robi.svg)](https://badge.fury.io/py/robi)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.7](https://img.shields.io/badge/python-3.7--3.10-blue)](https://www.python.org/downloads/release/python-360/)

<h1>
  <img align="right" height="250" src="./img/logo.png">
   <br> ROBI: Robust and Optimized Biomarker Identifier
</h1>

ROBI is a selection pipeline that select predictive biomarkers from any set of features.
The selection if performed with a robust and adjustable control of the number of false positives as well as a
control for confounders.

ROBI can control for confounders and already known biomarkers in order to select only new and relevant information.

## Installation

```shell
pip install robi
```

Although [PyTorch](https://pytorch.org/get-started/locally/) is not required to use the package, ROBI runs much faster
with its PyTorch implementation. The speed gain is great on CPU, and much greater on GPU.
To use the PyTorch implementation, simply install PyTorch, and ROBI will use it automatically.
To tell ROBI to use the GPU, simply set `device='cuda'` in the `make_selection` function.

## Utilisation


#### Basic usage
```shell
import robi

selection, scores = robi.make_selection(df,
                                        candidates,
                                        targets = 'outcome')
```

#### Number of false positive control

#### Control for confounders

#### Censored target

#### Multiple targets

You can find example notebooks in the `notebooks` folder of this repository.

radiomic DLBCL
TCGA
synthetic data


## Pipeline diagram


## Author

Louis Rebaud: [louis.rebaud@gmail.com](mailto:louis.rebaud@gmail.com)


## License

This project is licensed under the Apache License 2.0 - see the LICENSE.md file for details
