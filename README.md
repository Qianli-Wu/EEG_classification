# EEG_classification
This repository contains the final project for ECE C147/C247, which evaluates the performance of CNN + Transformer and CNN + GRU + SimpleRNN models on an EEG dataset. The objective is to classify subjects' movements using 22 channels of EEG electrode data. The repo focuses on providing the model architectures used in the project. Please note that the dataset and the project write-up are not included in this repository.


## Requirements
> [Tensorflow gpu acceleration](https://www.tensorflow.org/install/gpu?hl=zh-cn) does not support CUDA toolkit 11.2 or above
```
CUDA version <= 11.2
```

It is recommended to create a virtual environment for this project by conda
```
conda create -n egg_classification python=3.9
```
Then activate the environment by 
```
conda activate egg_classification
```
Then use `pip` to install all packages:
```
pip3 install -r requirements.txt
```
## Running
Here's an example:
```
$ python main.py --model=cnn+transformer --epoch=200 --learning_rate=4e-4 --num_heads=2 --ensemble=1
```
- **model**: model can be selected from `cnn`, `cnn+transformer`, `cnn+rnn`, and `transformer`.
- **epoch**: Integer. Number of epochs to train the model. 
- **learning_rate**: Floating point value.
- **num_heads**: Interger. Number of heads in transformer Multi-Head Attention Layer.
- **ensemble**: Integer. Number of models in model ensembling
- **patience**: Integer. Number of epochs that produced the monitored quantity with no improvement after which training will be stopped

## Transformers
* transformer needs larger dataset and more complex model to achieve similar performance with CNNs and RNNs since it makes less assumption on our model. 
  * The cost is its ability to extract information from data
* transformer requires similar complexity per layer but less Sequential Operations than RNNs and smaller Max Path Length than RNNs and CNNs

## Reference
> [Keras](https://keras.io)

> [BCI Competition 2008 – Graz data set A](https://www.bbci.de/competition/iv/desc_2a.pdf)

> `CNN with data processing.ipynb`

