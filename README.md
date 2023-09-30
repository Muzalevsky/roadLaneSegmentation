# Road Lane Segmentation

<div align="center">

[![PythonSupported](https://img.shields.io/badge/python-3.9-brightgreen.svg)](https://python3statement.org/#sections50-why)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)


![pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)

This project focuses on the segmentation of city road lane markings using deep machine learning techniques.

![DemoGIF](demo/test_camera_6(2).gif)

## Trained Models

</div>

* **The latest**: `best-valid-iou_1c07833020cf45939f24b5bec2bada5a.pth`
* **Trained in T204**: `best-valid-iou_40446a2b3b7543c292301a3b2da1ed67.pth`

> T204 trained on datasets with errors

<div align="center">

## Metrics

| Class label |         Class name         | IoU - The latest | IoU - T204 | The latest DICE | T204 DICE |
| :---------: | :------------------------: | :--------------: | :--------: | :-------------: | :-------: |
| background  |         background         |      0.998       |   0.998    |      0.999      |   0.998   |
|     SYD     | solid yellow dividing line |      0.986       |   0.823    |      0.987      |   0.899   |
|     BWG     | broken white guiding line  |      0.693       |   0.568    |      0.775      |   0.682   |
|     SWD     | solid white dividing line  |      0.794       |   0.617    |      0.881      |   0.727   |
|     SWS     |   solid white stop line    |      0.114       |   0.432    |      0.162      |   0.54    |
|    CWYZ     |         crosswalk          |      0.386       |    0.56    |      0.496      |   0.687   |
|      -      |             -              |        -         |     -      |        -        |     -     |
|    micro    |                            |      0.995       |   0.995    |      0.998      |   0.998   |
|    macro    |                            |      0.662       |   0.667    |      0.717      |   0.756   |


## Articles

</div>

A. R. Muzalevskiy, E. V. Serykh, M. M. Kopichev, E. V. Druian and M. A. Chernyshev, "Lane Marking Semantic Segmentation Using Convolutional Neural Networks," 2023 XXVI International Conference on Soft Computing and Measurements (SCM), Saint Petersburg, Russian Federation, 2023, pp. 123-126, doi: 10.1109/SCM58628.2023.10159034.

* [Lane Marking Semantic Segmentation Using Convolutional Neural Networks](https://ieeexplore.ieee.org/abstract/document/10159034), SCM, 2023


<div align="center">


## Repository contents

</div>

- [docs](docs) - documentation of the project
- [reports](reports) - reports generated (as generated from notebooks)
  > Check if you need to ignore large reports or keep them in Git LFS
- [configs](configs) - configuration files directory
- [notebooks](notebooks) - directory for `jupyter` notebooks
- [scripts](scripts) - repository service scripts
  > These ones are not included into the pakckage if you build one - these scripts are only for usage with repository
- [lane_detection_hackathon](lane_detection_hackathon) - source files of the project
- [.editorconfig](.editorconfig) - configuration for [editorconfig](https://editorconfig.org/)
- [.flake8](.flake8) - [flake8](https://github.com/pycqa/flake8) linter configuration
- [.gitignore](.gitignore) - the files/folders `git` should ignore
- [.pre-commit-config.yaml](.pre-commit-config.yaml) - [pre-commit](https://pre-commit.com/) configuration file
- [README.md](README.md) - the one you read =)
- [DEVELOPMENT.md](DEVELOPMENT.md) - guide for development team
- [Makefile](Makefile) - targets for `make` command
- [cookiecutter-config-file.yml](cookiecutter-config-file.yml) - cookiecutter project config log
- [poetry.toml](poetry.toml) - poetry local config
- [pyproject.toml](pyproject.toml) - Python project configuration
