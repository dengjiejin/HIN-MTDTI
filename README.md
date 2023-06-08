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
```
files:
  "pssm_arr": a protein vector with 220 length,
  "drug_arr": a drug fingerprint with 881 length,
  "int_ids": [p_id, d_id, 0 or 1], interaction label
```

## Data preprocessing
The preprocess file is [`processdata.py`](processdata.py). A directory `preporcess/` will be generated ,which contains 
preprocess adjacent matrix and train/val interaction index.


```
$ python ./processdata.py --dataset enzyme --crossval 1 --start_epoch 0 --end_epoch 2000 --common_neighbor 3 --adj_norm True --data_root ./data
```

## Training and Testing
Both training and testing programs can be implemented by scripts [`train.py`](train.py)

## Multi-task learning
```
    # if DTI task
    Loss = nn.BCELoss()
    # if DTA task
    Loss = nn.MSELoss()
```










