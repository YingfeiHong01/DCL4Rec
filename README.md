We use the recbole framework to facilitate our experiments. Our model is in the [sequential recommender](./recbole/model/sequential_recommender) folder and named as dcl4rec.py. 

## Installation

Recbole works with the following operating systems:

* Linux
* Windows 10
* macOS X

Recbole requires Python version 3.7 or later.

RecBole requires torch version 1.7.0 or later. If you want to use RecBole with GPU,
please ensure that CUDA or cudatoolkit version is 9.2 or later.
This requires NVIDIA driver version >= 396.26 (for Linux) or >= 397.44 (for Windows10).

### Install from source
```bash
https://github.com/YingfeiHong01/DCL4Rec && cd DCL4Rec
pip install -r requirements.txt
```

## Quick-Start
```bash
python run_dcl4rec.py
```

This script will run the DCL4Rec model on the ml-1m dataset.
