# Dense Subgraph Detection for Crowdsourcing with Arbitrary Adversaries


## Installation
To install the package, first create a virtual environment with a Python version at least `3.12`. With `conda`, this can be done by: 
```bash
$ conda create -n venv -c conda-forge python=3.12
```
Then, in a terminal `cd`ed into the project directory, perform an installation from a local directory after activating created environment: 
```bash
$ pip install .
```

## Usage

Two experiments on the effect of number of adversaries and targeted points are implemented and can be found under `scripts` folder. 
Both experiments are callable from terminal as follows: 
```bash
$ python scripts/exp_n_adversaries.py run --dataset rte --adv-frac 0.3 --aggregator wds
$ python scripts/exp_n_targeted.py run --dataset rte --target-frac 0.3 --aggregator wds
```
Further details about command line options can be found by
```bash
$ python scripts/exp_n_adversaries.py run --help
$ python scripts/exp_n_targeted.py run --help
```

## Data

The scripts implemented to work on four datasets: rte, sp, web and dog. These need to be downloaded to run the scripts. rte dataset can be 
downloaded as
```bash
>>> from pathlib import Path
>>> import aad
>>> root_dir = Path("_data", "inputs")
>>> response_mat, gt_labels = datasets.read_rte(root_dir, download=True)
```
Once this is done the scripts can be applied to rte dataset. The last three can be installed from literature as `.mat` file, which 
is the required file format for the scripts.
