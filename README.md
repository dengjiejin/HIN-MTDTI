## Required Packages

* Python 3.8.0 
* Pytorch 1.7.0
* numpy 1.23.1
* Pytorch_geometric 2.1.0
* scikit-learn 0.21.3
* tqdm 4.64.1 


## Experimental Datasets

Please see datasets in [`data/`](data/) with detailed instructions.
original data contains protein vectors, drug fingerprints and interaction labels

## Data preprocessing
The preprocess file is [`processdata.py`](processdata.py). A directory `preporcess/` will be generated ,which contains 
preprocess adjacent matrix and train/val interaction index.

## Training and Testing
Both training and testing programs can be implemented by scripts [`train.py`](train.py)










