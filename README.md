![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-experimental-orange)

# From-Scratch Multi-Layer Perceptron (MLP)

**Author:** B. Darmendrail\
**Language:** Python\
**Libraries:** numpy, numba, matplotlib, scikit-learn, tqdm  

---

## Project Overview

This project was developed as part of a personal exploration of neural networks. The goal is to build a multilayer perceptron (MLP) entirely from scratch in Python, in order to better understand the inner workings of neural networks. Without claiming to achieve a certain level of performance, the aim is to learn through implementation, exploring the dynamics of training and the behavior of parameters in simple architectures.
The implementation follows an object-oriented design, with core components including:

- **Network** – main MLP structure  
- **NeuralLayer** – definition of individual layers

Numba is used to speed up some numerical calculations. According to my estimates, the gain is around 10%.

---

## Main Features

### MLP Training
- Create and train an MLP on small datasets.
- Useful for testing hypotheses about learning behavior in simple networks.
- Observe learning curves and decision boundaries.

### Statistical Analysis of Multiple Networks
- For fixed hyperparameters, train a number of MLPs and visualize trends in L2 norms of weights and/or biases.

### Performance Mapping
- For a chosen architecture, generate a performance map showing outcomes as a function of learning rate and iteration count.  
- Provides an intuitive visualization of sensitivity to hyperparameters and possible optimal regions.  

---

## Project Structure
```
MLP-from-scratch/
│
├─ src/
│ ├─ neuron_class.py        # Classes: Network, NeuralLayer, NetworkConfig
│ ├─ network_trainer.py     # Training routines and functions
│ ├─ network_visualizer.py  # Visualization functions
│ ├─ network_config.py      # generic settings for an MLP
│ |─ numba_functions.py     # Optimized routines with numba
│ └─ dataset_creation.py    # Datasets generation
│
├─ examples/ # Usage examples and Notebook
├─ requirements.txt
├─ README.md
└─ .gitignore
```
---

## Limitations

- Restricted to shallow networks due to hardware constraints.  
- Many advanced techniques (e.g., batch training, large-scale datasets) are not included.  
- Statistical observations are based on small datasets and simplified architectures.  

---

## Notes on Statistics

- The norms of weights and biases evolve in line with intuition: weight magnitudes tend to increase with more training iterations, while higher regularization coefficients help reduce them.  
- When plotting the L2 norm of biases against the L2 norm of weights, certain unexpected patterns appear. The distribution of solutions in weight–bias space shows structures that do not seem purely random, though their interpretation remains open.  
- Performance maps provide a practical tool for visualizing training dynamics and assessing the sensitivity of the network to different hyperparameters.  

---

## Acknowledgments

The early steps of this project were inspired by the YouTube channel **machine_learnia**.  
