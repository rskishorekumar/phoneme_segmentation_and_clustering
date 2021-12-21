# phoneme_segmentation_and_clustering

We have developed another method for detecting the desired keyword / phrase candidate matched between the pair of speech utterances. The input to out method is the posterior features extracted for the speech utterances and the method outputs matching time stamps information between a pair of speech utterance. With this matching information between the speech utterances the clustering of speech utterances is done. This clustering of speech utterances groups the utterances at some broader domains at semantic level. For example if the speech repository consist of news bulletins, the output clusters representing the broader domains such as politics, social, sports and weather.

The above source codes are implemented in a manuscript titled "Unsupervised Pattern Discovery and
Clustering of Speech Utterances based on Phoneme Segmentation" by Kishore Kumar R, Vineeth Chinmay, K Sreenivasa Rao submitted.

More details are provided in the above mentioned paper.

usage:

(i) Load the data_phoneme_features.mat file. This file contains the posterior features of two speech utterances and each vector represents a phoneme unit. 

(ii) run the matlab program Keyword_detection.m and the last section in the program will give the matching information between a pair of speech utterances. 

Source code is tested in Matlab 2016B, Linux Environment.

# Citation 

If you use the code for research work, please cite:
```
@article{R20211,
title = {Phoneme Segmentation-Based Unsupervised Pattern Discovery and Clustering of Speech Signals},
journal = { Circuits Systems and Signal Process},
year = {2021},
doi = {https://doi.org/10.1007/s00034-021-01876-6},
url = {https://link.springer.com/article/10.1007/s00034-021-01876-6},
author = {Kishore Kumar R and Sreenivasa Rao K}
}

```
