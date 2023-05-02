<div align="center">

<img src="https://github.com/AsakoKabe/AdeleCV/blob/main/docs/logo.png?raw=true" alt="drawing" width="200"/>

**Auto DEap LEarning Computer Vision**

**Python library and dashboard for hyperparameter search and model training for computer vision tasks
based on [PyTorch](https://pytorch.org/), [Optuna](https://optuna.org/),
    [FiftyOne](https://docs.voxel51.com/), [Dash](https://dash.plotly.com/),
    [Segmentation Model Pytorch](https://github.com/qubvel/segmentation_models.pytorch).**  

[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/AsakoKabe/AdeleCV/blob/main/LICENSE)
[![Read the Docs](https://img.shields.io/readthedocs/smp?style=for-the-badge&logo=readthedocs&logoColor=white)](https://adelecv.readthedocs.io/en/latest/) 
[![GitHub Workflow Status (branch)](https://img.shields.io/github/actions/workflow/status/AsakoKabe/AdeleCV/code-style.yaml?branch=main&style=for-the-badge)](https://github.com/AsakoKabe/AdeleCV/actions/workflows/code-style.yaml)

[![PyPI](https://img.shields.io/pypi/v/adelecv?color=blue&style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/adelecv/) 
[![PyPI - Downloads](https://img.shields.io/pypi/dm/adelecv?style=for-the-badge&color=blue)](https://pepy.tech/project/adelecv) 
<br>
</div>

The main features of this library are:

 - Fiftyone dataset integration with prediction visualization
 - Uploading your dataset in one of the popular formats, currently supported - 2
 - Adding your own python class for convert dataset
 - Displaying training statistics in tensorboard
 - Support for all samples from optuna
 - Segmentation use smp: 9 model architectures, popular losses and metrics, see [doc smp](https://github.com/qubvel/segmentation_models.pytorch)
 - Convert weights to another format, currently supported - 1 (onnx)
 
### [📚 Project Documentation 📚](https://adelecv.readthedocs.io/en/latest/)

Visit [Read The Docs Project Page](https://adelecv.readthedocs.io/en/latest/) or read following README to know more about Auto Deap Learning Computer Vision (AdeleCV for short) library

### 📋 Table of content
 1. [Examples](#examples)
 2. [Installation](#installation)
 3. [Instruction Dashboard](#instruction-dashboard)
 4. [Architecture](#architecture) 
 5. [Citing](#citing)
 6. [License](#license)


### 💡 Examples <a name="examples"></a>
 - Example api [notebook](https://github.com/AsakoKabe/AdeleCV/blob/main/example/api.ipynb)
 - See [video](https://www.youtube.com/watch?v=3kztXbAnkYg&ab_channel=DenisMamatin) on the example of using dashboard

### 🛠 Installation <a name="installation"></a>
Install torch cuda if not installed:
```bash
$ pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

PyPI version:
```bash
$ pip install adelecv
````
Poetry:
```bash
$ poetry add adelecv
````

### 📜 Instruction Dashboard <a name="instruction-dashboard"></a>
1. Create .env file. 

See [docs](https://adelecv.readthedocs.io/en/latest/config.html). 

Notification_LEVEL: DEBUG | INFO | ERROR

Example:
```
TMP_PATH='./tmp'
DASHBOARD_PORT=8080
FIFTYONE_PORT=5151
TENSORBOARD_PORT=6006
NOTIFICATION_LEVEL=DEBUG
```

2. Run (about 30 seconds (I'm working on acceleration)).
```bash
adelecv_dashboard --envfile .env
```

3. Help
```bash
adelecv_dashboard --help
```


### 🏰 Architecture <a name="architecture"></a>
![architecture](https://github.com/AsakoKabe/AdeleCV/blob/main/docs/architecture.png?raw=true)

The user can use the api or dashboard(web app). 
The api is based on 5 modules:
- data: contains an internal representation of the dataset, classes for converting datasets, fiftyone dataset
- _models: torch model, its hyperparams, functions for training
- optimize: set of hyperparams, optuna optimizer
- modification model: export and conversion of weights
- logs: python logging 

The Dash library was used for dashboard. It is based on components and callbacks on these component elements.

### 📝 Citing
```
@misc{Mamatin:2023,
  Author = {Denis Mamatin},
  Title = {AdeleCV},
  Year = {2023},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/AsakoKabe/AdeleCV}}
}
```

### 🛡️ License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/AsakoKabe/AdeleCV/blob/main/LICENSE)
