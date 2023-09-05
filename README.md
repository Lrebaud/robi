[![PyPI version](https://badge.fury.io/py/robi.svg)](https://badge.fury.io/py/robi)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.7](https://img.shields.io/badge/python-3.7--3.10-blue)](https://www.python.org/downloads/release/python-360/)
<p>
  <img align="right" height="290" src="./img/logo.png">
</p>

# ROBI<br>Robust and Optimized Biomarker Identifier

<p align="justify">
    ROBI is a selection pipeline that select predictive biomarkers from any set of features.
    The selection if performed with a robust and adjustable control of the number of false positives as well as a
    control for confounders.
    ROBI can control for confounders and already known biomarkers in order to select only new and relevant information.
</p>

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

First, ROBI must be imported:
```python
import robi
```

Then, a pandas dataframe need to be defined were each row is a patient, and each column a feature
(biomarker, outcome, ...), such as:

| candidate_1 | candidate_2 | outcome |
|-------------|-------------|---------|
| 0           | 100         | 10      |
| 0.1         | -2          | 25      |

with `candidate_1` and `candidate_2` the candidate biomarkers that we want to evaluate and `outcome` the target
(e.g. the feature that we want to be predicted by the selected biomarkers).

Then, the selection can be performed with:
```python
selection, scores = robi.make_selection(df,
                                        candidates = ['candidate_1', 'candidate_2'],
                                        targets = 'outcome')
```


#### Number of false positive control


![](C:\Users\louis\Documents\work\articles\article_pipeline\robi\img\selection_plot.png)


#### Control for confounders

If confounders are in the dataset, they can be listed in the `confounders` parameter. ROBI will discard any candidate
that is sensitive to these confounders, making sure that any selected biomarker is relevant and worth studying further.
```python
selection, scores = robi.make_selection(df,
                                        candidates,
                                        targets = 'outcome',
                                        confounders = ['age', 'sex'])
```
This way, any candidate whose hazard ratio changes by more than 10% when confounders are introduced in a Cox model,
will be discarded.

#### Control for known biomarkers

If some biomarkers are already known and used, we can avoid selecting candidates that are simply replicating this known
information. For instance, if we know that tumor volume affect the outcome of patients, we can specify
`known = ['tumor_volume']`such as:
```python
selection, scores = robi.make_selection(df,
                                        candidates,
                                        targets = 'outcome',
                                        known = ['tumor_volume'],
                                        confounders = ['age', 'sex'])
```
This way, any candidate that is simply a proxy of the tumor volume will be discarded. Multiple known biomarkers can be
listed. Collinearity and multicollinearity will be tested.

#### Censored target

ROBI can handle censored targets (e.g. we know that a patient was alive until a certain date, but then we don't if
he died or when). For instance, to use the Overall Survival (OS), one must specify:
```python
selection, scores = robi.make_selection(df,
                                        candidates,
                                        targets = {
                                            'OS': ('OS_time', 'OS_happened')
                                        })
```
with `OS_time` being the time between diagnosis and death or end of study, and `OS_happened` a boolean feature 
stating if a patient died (True or 1) or not (False or 0) during the study.

#### Multiple targets

ROBI can perform the biomarker selection for multiple targets at the same time. For instance, the candidates could be
evaluated for OS and Progression Free Survival (PFS). Simply pass them to the `targets` parameter as a dictionary:
```python
selection, scores = robi.make_selection(df,
                                        candidates,
                                        targets = {
                                            'PFS': ('PFS_time', 'PFS_happened'),
                                            'OS': ('OS_time', 'OS_happened')
                                        })
```
The key of the dictionary is the name of the target. The first element of the tuple is the time, the second says if
the event happened or not.

When giving multiple targets, some could be censored while other might be uncensored. Give them like in the following
example:
```python
selection, scores = robi.make_selection(df,
                                        candidates,
                                        targets = {
                                            'uncensored_target': ('uncensored_target'),
                                            'censored_target': ('censored_target_time', 'censored_target_happened')
                                        })
```

## Examples
You can find example notebooks in the `notebooks` folder of this repository.

radiomic DLBCL
TCGA
synthetic data


## Pipeline diagram


## Author

Louis Rebaud: [louis.rebaud@gmail.com](mailto:louis.rebaud@gmail.com)


## License

This project is licensed under the Apache License 2.0 - see the LICENSE.md file for details
