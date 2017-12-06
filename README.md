# Generating Visual Representations for Zero-Shot Classification 
This paper addresses the task of learning an image classifier when some categories are defined by semantic descriptions only (\eg visual attributes) while the others are defined by exemplar images as well. This task is often referred to as the Zero-Shot classification task (ZSC). Most of the previous methods rely on learning a common embedding space allowing to compare visual features of unknown categories with semantic descriptions. This paper argues that these approaches are limited as i) efficient discriminative classifiers can't be used ii) classification tasks with seen and unseen categories (Generalized Zero-Shot Classification or GZSC) can't be addressed efficiently. In contrast, this paper suggests to address ZSC and GZSC by i) learning a conditional generator using seen classes ii) generate artificial training examples for the categories without exemplars. ZSC is then turned into a standard supervised learning problem. Experiments with 4 generative models and 5 datasets experimentally validate the approach, giving state-of-the-art results on both ZSC and GZSC.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for research purposes.

## Prerequisites

This code requires a Python 2.7 environment.

This code is based on Tensorflow +1.0. Please follow the official instructions to install [Tensorflow](https://www.tensorflow.org/install/)  in your environment.

Remaining dependencies are:
```
Python 2.7
Tensorflow +1.0
Numpy +1.13
Sklearn +0.19
scipy +0.19
```


## Datasets
* AWA1: [Animals with Attributes 1](https://cvml.ist.ac.at/AwA/)  
* AWA2: [Animals with Attributes 2](https://cvml.ist.ac.at/AwA2/) 
* CUB: [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)  
* APY: [aYahoo & aPascal](http://vision.cs.uiuc.edu/attributes/)  
* SUN: [SUN Attribute](https://cs.brown.edu/~gen/sunattributes.html) 

