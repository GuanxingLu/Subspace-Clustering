# Subspace-Clustering
The implementation of our paper *Semi-Supervised Subspace Clustering via Tensor Low-Rank Representation*
https://arxiv.org/abs/2205.10481

This repository contains:

1. [Datasets and Selected Annotations](data) in our paper, includeing ORL, YaleB, COIL20, Isolet, MNIST, Alphabet, BF0502 and Notting-Hill.
2. A [comparision demo](demo_parallel.m).
3. A [function](tlrr_tnn_new.m) to implement the proposed method.
4. Some raw experimental [results](result).
5. A [visualization demo](Visualization_demo_parallel.m) of the result files.

## Usage

Before running the code, you need to download the following toolbox:
1. LinADMM library from: https://github.com/canyilu/LibADMM
2. Graph Signal Processing Toolbox (GSPBox) from: https://github.com/epfl-lts2/gspbox
3. ClusteringMeasure from: https://github.com/jyh-learning/MVSC-TLRR

Any questions, please contact me through the Email guanxing at seu dot edu dot cn
