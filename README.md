# EEG_classification
The final project for ECE C147/C247. This project evaluates CNN + Transformer and CNN + LSTM on EEG dataset. The goal is to classify the movement of the subjects by 22 channels of EEG electrodes. 


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

## Transformers
* transformer needs larger dataset and more complex model to achieve similar performance with CNNs and RNNs since it makes less assumption on our model. 
  * The cost is its ability to extract information from data
* transformer requires similar complexity per layer but less Sequential Operations than RNNs and smaller Max Path Length than RNNs and CNNs

## Reference
> `CNN with data processing.ipynb`

> `CNN-LSTM Hybird with data processing.ipynb`

> keras
