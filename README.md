# A Neural Network Approach For The Propensity  Modelling of Online Conversion Using Webpage Visit Information

This repository contains the prototype Keras/TensorFlow implementation for the paper 'A Neural Network Approach For The Propensity  Modelling of Online Conversion Using Webpage Visit Information' by Johannes De Smedt, Ewelina Lacka, Spyro Nita, Andrea Rosales, Hans-Helmut Kohls, and Ross Paton.

* [read_ctr_data.py](read_ctr_data.py) contains code to pre-process the [Avazu click-through dataset](https://www.kaggle.com/c/avazu-ctr-prediction/data) which serves as a benchmark.
* [nn_model_attribution.py](nn_model_attribution.py) contains the recurrent neural network with time attention layer for time between web page visits/sessions, embeddings for device (not that location is not available for the Avazu dataset) and channels, as well as a convolution for web page visited within a single session.
