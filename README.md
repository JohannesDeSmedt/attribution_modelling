# A Neural Network Approach For The Propensity  Modelling of Online Conversion Using Webpage Visit Information

This repository contains the prototype Keras/TensorFlow implementation for the paper 'A Neural Network Approach For The Propensity  Modelling of Online Conversion Using Webpage Visit Information' by Johannes De Smedt, Ewelina Lacka, Spyro Nita, Andrea Rosales, Hans-Helmut Kohls, and Ross Paton.

For more info, contact the corresponding author [Johannes De Smedt](mailto:jdesmed@ed.ac.uk).

## Parameters
Note that there are two parameters to look out for:
* The minimum number of visits per user/customer (min_vis)
* The cutoff or number of time steps per user (which pads or shortens user session sequences accordingly)

## Files
* [read_ctr_data.py](read_ctr_data.py) contains code to pre-process the [Avazu click-through dataset](https://www.kaggle.com/c/avazu-ctr-prediction/data) which serves as a benchmark.
* [nn_model_attribution.py](nn_model_attribution.py) contains the recurrent neural network with time attention layer for time between web page visits/sessions, embeddings for device (not that location is not available for the Avazu dataset) and channels, as well as a convolution for web page visited within a single session.
* [data.csv](data.csv) contains pre-processed files based on the Avazu datasets for a minimum visit of 2, 3, and 5 and a cutoff of 3, 3, and 5 respectively. The first number signifies min_vis and the second the cutoff.
