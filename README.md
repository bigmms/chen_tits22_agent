# Automatic Itinerary Planning Using Triple-Agent Deep Reinforcement Learning

![](https://img.shields.io/badge/Tensorflow-1.14.0-yellow)
![](https://img.shields.io/badge/Cuda-10.0-blue)

![](save_graph/main_network.png)


This repository provides the dataset and code for the following paper:

**Paper**: Automatic Itinerary Planning Using Triple-Agent Deep Reinforcement Learning

**Authors**: Bo-Hao Chen, Jin Han, Shengxin Chen, Jia-Li Yin, Zhaojiong Chen

in [IEEE TITS](https://ieeexplore.ieee.org/document/9766177).


## Abstract
Automatic itinerary planning that provides an epic journey for each traveler is a fundamental yet inefficient task. Most existing planning methods apply heuristic guidelines for certain objective, and thereby favor popular preferred point of interests (POIs) with high probability, which ignore the intrinsic correlation between the POIs exploration, traveler's preferences, and distinctive attractions. To tackle the itinerary planning problem, this paper explores the connections of these three objectives in probabilistic manner based on a Bayesian model and proposes a triple-agent deep reinforcement learning approach, which generates 4-way direction, 4-way distance, and 3-way selection strategy for iteratively determining next POI to visit in the itinerary. Experiments on five real-world cities demonstrate that our triple-agent deep reinforcement learning approach can provide better planning results in comparison with state-of-the-art multiobjective optimization methods.

## Requirements

### Dependencies
* cuda 10.0
* cudnn 7.3.1
* Python 3.5
* xlrd==1.2.0
* geopy==1.20.0
* pymoo==0.3.2
* requests==2.22.0
* tensorflow==1.14.0
* matplotlib==3.1.1
* numpy==1.16.1
* autograd==1.3
* scikit_learn==0.21.2

### Dataset
The train image dataset can be downloaded from [here](https://pan.baidu.com/s/1pWibbJ9f5XpSFhwPR66A3g)(extraction code:vf22).

### It was tested and runs under the following OSs:
* Windows 10
* Ubuntu 16.04

Might work under others, but didn't get to test any other OSs just yet.

## Getting Started:
### Usage
* Training
```bash
$ python train.py --place=your_dataset_type
```
* Testing
```bash
$ python test.py --place=your_dataset_type --iniloc=starting_point_coordinates --endloc=terminal_point_coordinates --tottime=total_number_of_locations_traveled --Budgetlevel=your_budget_level --mytype=preferred_location_type
```

## Results
![](save_graph/LA.png)
![](save_graph/London.png)

The images were the recovered results using the methods of (a) Engin et al., (b) Li et al., (c) Qian et al., (d) Yang et al., (e) Wei et al., and (f) Zhang et al. as well as (g) our previous method and (h) the method proposed in this study. Please refer to our [paper](https://ieeexplore.ieee.org/document/9357944) for more information.

## License + Attribution
This code is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Commercial usage is not permitted. If you use this code in a scientific publication, please cite the following [paper](https://ieeexplore.ieee.org/document/9357944):
```
@ARTICLE{ChenTITS2021,
  author={B. -H. {Chen} and S. {Ye} and J. -L. {Yin} and H. -Y. {Cheng} and D. {Chen}},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Deep Trident Decomposition Network for Single License Plate Image Glare Removal}, 
  year={2021},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TITS.2021.3058530}}
```
