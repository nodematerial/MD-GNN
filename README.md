# Prediction of potential energy profiles of molecular dynamic simulation by graph convolutional networks

[![GitHub license](https://img.shields.io/github/license/nodematerial/MD-GNN?style=flat-square)](https://raw.githubusercontent.com/nodematerial/MD-GNN/main/LICENSE)


This is the implementation for [Prediction of potential energy profiles of molecular dynamic simulation by graph convolutional networks](https://www.sciencedirect.com/science/article/pii/S0927025623004421?via%3Dihub)

<p align="center"><img src="https://github.com/nodematerial/MD-GNN//blob/main/.github/graphical_abstract.png?raw=true" width="600"/></p>

MD-GNN (GCN-based Predictor for mesoscale metallic systems) is constructed to predict physical properties from trajectory of MD simulation.

<p align="center"><img src="https://github.com/nodematerial/MD-GNN//blob/main/.github/pic5.png?raw=true" width="600"/></p>

## Table of contents

- [Environment Setup](#environment-setup)
  - [Required hardware specifications](#required-hardware-specifications)
  - [Prerequisites](#prerequisites)
  - [Cloning repository](#cloning-repository)
  - [Docker_Settings](#docker-settings)
- [Usage](#usage)
  - [Directory Tree](#directory-tree)
  - [Avaliable Commands](#avaliable-commands)
    - [Step1 Execute MD Simulation by using LAMMPS](#step1-execute-md-simulation-by-using-lammps)
    - [Step2 Making Graph Representations](#step2-make-graph-representations)
    - [Step3 Prepare other data](#step3-prepare-other-data)
    - [Step4 Train Machine Learning Model](#step4-train-machine-learning-model)
    - [Configure Settings(config.yml)](#configure-settings)
    - [Step5 Predict potential energy (inference)](#step5-predict-potential-energy-inference)
- [Citation](#citation)
- [Contact](#contact)


## Environment Setup

If you want to use the same environment used in the paper, you can follow instructions below.

### Required hardware specifications

To perform calculations using the source code in this repository, the following environment is required

* NVIDIA GPU that is compatible with CUDA
* CPU, memory, and auxiliary storage devices with sufficient specifications to perform scientific and technical calculations

### Prerequisites

1. Docker
2. NVIDIA Drivers
3. NVIDIA Container Toolkit
   - NVIDIA Container Toolkit is a software that enables the use of NVIDIA GPUs within containerized environments
   - You can find how to install NVIDIA container toolkit on [this page](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Cloning repository
clone this repository by executing the command below.

```bash
git clone git@github.com:nodematerial/MD-GNN.git
```

### Docker Settings

1. Build Docker Image by executing the command below.
```bash
docker-compose build
```

2. Run Docker Container by executing the command below.
```bash
docker-compose up -d
```

3. Attach to the running container by executing the command below.
```bash
docker attach MD-GNN [YOUR CONTAINER ID]
```

4. install Python modules by using [Poetry](https://github.com/python-poetry/poetry)
```bash
poetry install
```


## Usage

### Directory_Tree

```bash
├── LICENSE
├── README.md
├── compose.yml
├── configs          # Config file to Create Input Data for ML model
├── dataset          # Input Data for ML model (Graphs, Features, Labels)
├── docker           # Directory to put Dockerfile
├── dumpfiles        # Directory to put Dumpfile (atomic information by MD simulation)
├── lmpfile          # Directory to put setting file for LAMMPS
├── makefile
├── poetry.lock
├── pyproject.toml
├── script          # Directory to put script for Machine Learning experimet
└── shell_script    # Directory to put task definitions
```

### Avaliable_Commands

#### Step1 Execute MD Simulation by using LAMMPS

LAMMPS[1] (Large-scale Atomic/Molecular Massively Parallel Simulator) is an open-source molecular dynamics software package designed to simulate and study the behavior of materials at the atomic and molecular scale.

1. You have to define MD simulation tasks according to the notation described in [documentation](https://docs.lammps.org/Manual.html).
2. You can start an MD simulation by executing the following command
```bash
 lmp -sf gpu -in [YOUR LAMMPS FILE]
```
- or you can excecute defined task by command below
  - You have to define two parameters in config file(.yml) below `config` directory
    - lmpfile_name: The name of LAMMPS file
    - dir_name: Relative path where lmpfile file is put
```bash
 make lmp CONF_NAME=[YOUR_CONFIG_FILE.yml]
```

[1] S.J. Plimpton, Fast parallel algorithms for short-range molecular dynamics, J. Comput.
Phys., 117 (1995) 1.

#### Step2 Make Graph Representations
To use GCN, it is essential to convert atomic coordinates of the system into the graph
structure. You can convert atomic information in dumpfiles into Graphs and Features by following commands below

- You have to define four parameters in config file (.yml) below `config` directory
  - dirname: Relative path where dumpfile is put
  - cutoff: cutoff value(angstrom), which is the threshold to evaluate whether two nodes is connected. (The edge is created
between two atoms when the interatomic distance is shorter than the threshold distance.)
  - time: determine the timestep to use, three elements of the list represents [ time of first step, time of last step, interval of each steps ] (unit: fs)
  - division: Specifies the number of divisions of the system into cells, specified as a list with three elements, each of which refers to the number of divisions in the x, y, and z directions.
```bash
 make makegraph CONF_NAME=[YOUR_CONFIG_FILE.yml]
```

#### Step3 Prepare other data
Before starting the training of ML models, you need to prepare supplemantal data, concretely, you have to run command
that excecute Common Neighbor Analysis and convert lammps logfile into csv data


- You have to define four parameters in config file(.yml) below `config` directory
  - dirname: Relative path where dumpfile is put
  - time: determine the timestep to use, three elements of the list represents [ time of first step, time of last step, interval of each steps ] (unit: fs)
```bash
 make cna CONF_NAME=[YOUR_CONFIG_FILE.yml]
 make thermo CONF_NAME=[YOUR_CONFIG_FILE.yml]
```

#### Step4 Train Machine Learning Model
You can train the ML model by runnning command below, 

```bash
poetry run python3 train.py
```

Detailed setting is described in `config.yml` which is placed in each experiment folder.

#### Configure Settings

```yml
cutoff : [String] # The cutoff which is used as the threshold to convert atoms position into graphs
interval: [Integer] # The interval of timestep
data_per_condition: [Integer] # the amount of data for each condition
train_dirs: [List[String]] # List of directory name in which the training data is placed
valid_dirs: [List[String]] # List of directory name in which the validation data is placed
all_dirs:  [List[String]] # List of directory name in which the training&validation data is placed

seed: [Integer] # random seed
hidden_channels: [Integer] #dimension of hidden layer of GCN convolution
epochs: [Integer] # The number of epoch of training.
lr: [Float] # The Learning Rate of training.

wandb: [boolean] # If you can use Weight & Biases (https://wandb.ai/home), you can manage the training result by Wandb.
project_name: [String] # Project name of experimet (this name is used as namespace in Wandb)
exp_name: [String] # Experiment name

BASE_ENERGY: [Float] # Potential energy at the first step (i.e. 0sec)
```

#### Step5 Predict potential energy (inference)

you can predict the potential energy by excecuting command below.
The prediction result will be saved as `prediction.csv` and visualization results will be placed in `images` folder


```bash
poetry run python3 infer.py
poetry run python3 mapping.py
```

## Citation

```
@article{NODA2023112448,
  title = {Prediction of potential energy profiles of molecular dynamic simulation by graph convolutional networks},
  journal = {Computational Materials Science},
  volume = {229},
  pages = {112448},
  year = {2023},
  issn = {0927-0256},
  doi = {https://doi.org/10.1016/j.commatsci.2023.112448},
  url = {https://www.sciencedirect.com/science/article/pii/S0927025623004421},
  author = {Kota Noda and Yasushi Shibuta},
  keywords = {Molecular dynamics, Machine learning, Graph convolutional networks, Graph representation, Solid-liquid coexisting system},
}
```

## Contact
If you have some trouble, please contact us by opening Issue.
