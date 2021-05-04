## LSGNN
A PyTorch implementation for the CIC 2020 paper below:  
Discovering Localized Information for Heterogeneous Graph Node Representation Learning.  
Lin Meng<sup>*</sup>, Ning Yan, Masood Mortazavi, Jiawei Zhang.   
[\[paper\]](https://ieeexplore.ieee.org/document/9319009)

<sup>*</sup>This work is based on the summer intern in Futurewei Inc.

## Dataset
We test it on three datasets, the preprocessed datasets are in ./data/:  
- ACM: graph_acm_v3.pk,  labels_acm_v3.pk
- DBLP: graph_DBLP_four_area.pk, labels_DBLP_four_area.pk
- IMDB: graph_imdb.pk, labels_imdb.pk

All raw datasets are publicly accessible in the [page](https://github.com/Jhy1993/HAN/tree/master/data).

Our code reuses part of codes in [HGT](https://github.com/acbull/pyHGT)

## Requirements
Codes are written under python 3.7
- pytorch_geometric 1.6.1+
- pytorch 1.5.0+
- scipy
- numpy
- pandas
- skicit-learn
- dill 

If using docker, please pull [pytorch/pytorch:latest](https://hub.docker.com/r/pytorch/pytorch/) image and install the required packages.


## Run
The preprocessing codes are files with name starting with preprocess.

To run the code, simply use '''python preprocess_xxx.py'''


'''bash
    python train_node_classification.py --sample_width 6 --sample_depth 2 --dataset _acm_v3 --target_type paper --n_hid 8  > acm.out 

    python train_node_classification.py --sample_width 5 --sample_depth 2 --dataset _DBLP_four_area --target_type author --n_hid 4 > dblp.out

    python train_node_classification.py --sample_width 4 --sample_depth 2 --dataset _imdb --target_type movie --n_hid 8 > imdb.out 
'''

## Docker environment set up
If you prefer to use docker, please cd to the dockerfile folder /home/meng/code/isonode .
- RUN dockerfile to build the required image
    '''bash
        docker build --tag pygraph:1.0
    '''
- RUN a docker container
    '''bash
        docker run --gpus all -d --ipc=host -it pygraph:1.0  /bin/bash 
    '''

    In my machine, one existing image with all required packages is called pytorch/graph:v4.

- Then got the CONTAINER_ID by '''docker ps''' and enter the container via bash
    '''bash
        docker exec -it CONTAINER_ID /bin/bash
    '''

## Plot
The plot file is node_clustering.py

