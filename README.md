# SSC-TLRR
This is an implementation of our IEEE TCSVT 2023 paper:

[*Semi-Supervised Subspace Clustering via Tensor Low-Rank Representation*](https://arxiv.org/abs/2205.10481)<br />
Yuheng Jia, Guanxing Lu, Hui Liu, Junhui Hou<br />
Southeast University, Caritas Institute of Higher Education, City University of Hong Kong

![image](image/illustration.png)

This repository contains:

- [Datasets and Selected Annotations](data) in our paper, includeing ORL, YaleB, COIL20, Isolet, MNIST, Alphabet, BF0502 and Notting-Hill, and a matched visualization demo.
- A [Function](tlrr_tnn_new.m) to implement the proposed method.
- A [Comparision Demo](demo_parallel.m) of the mentioned methods (you may need to refer to possible official implementations, or implement them yourself) in our manuscript, including LRR, DPLRR, SSLRR, L-RPCA, CP-SSC, SC-LRR and CLRR.
- Some raw experimental [Results](result).
- A [Visualization Demo](Visualization_demo_parallel.m) of the result files.
- A [Dataset Visualization Demo](Visualization_dataset.m) to visualize the data.

## Usage

Before running the code, you need to download the following toolboxes:
- LibADMM library from: https://github.com/canyilu/LibADMM
- Graph Signal Processing Toolbox (GSPBox) from: https://github.com/epfl-lts2/gspbox
- Clustering Measure from: https://github.com/jyh-learning/MVSC-TLRR

## Errata

- We have added ``genWv3.m``, which is used to generate the $k$-NN graph from data.
- We have renamed the function ``Normalize_test`` (previously used as a copy of the ``normalize`` function for an older version of MATLAB) to ``normalize`` for convenience.
- We have added ``norm21.m`` to compute the objective value. This does not affect the training progress.

If you still encounter any problems during installation, please feel free to open an issue.

## Citation

If you find this repository useful, please consider citing our work:

```
@ARTICLE{10007868,
  author={Jia, Yuheng and Lu, Guanxing and Liu, Hui and Hou, Junhui},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Semi-Supervised Subspace Clustering via Tensor Low-Rank Representation}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2023.3234556}}
  ```

