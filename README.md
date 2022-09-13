
# Public code repo for Tufts CS 152 - Bayesian Deep Learning - in Fall 2022

Course website: <https://www.cs.tufts.edu/comp/152BDL/2022f/>

Cross-platform compatibility tests:   
[![MacOS status](https://github.com/tufts-ml-courses/cs152BDL-22f-assignments/actions/workflows/verify_macos.yml/badge.svg)](https://github.com/tufts-ml-courses/cs152BDL-22f-assignments/actions/workflows/verify_macos.yml)
• [![Ubuntu status](https://github.com/tufts-ml-courses/cs152BDL-22f-assignments/actions/workflows/verify_ubuntu.yml/badge.svg)](https://github.com/tufts-ml-courses/cs152BDL-22f-assignments/actions/workflows/verify_ubuntu.yml)
• [![Windows status](https://github.com/tufts-ml-courses/cs152BDL-22f-assignments/actions/workflows/verify_windows.yml/badge.svg)](https://github.com/tufts-ml-courses/cs152BDL-22f-assignments/actions/workflows/verify_windows.yml)


### Installation 

Three steps to install the BDL Python environment anywhere

1) Download and install miniconda

```
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
```

Then run that script (`bash Miniconda2-latest-Linux-x86_64.sh`).

2) Clone this repo

```
$ git clone https://github.com/tufts-ml/cs152BDL-22f-assignments.git
```

3) Create the conda environment

```
cd path/to/cs152BDL-22f-assignments/

conda env create -f bdl_2022f_env.yml
```

