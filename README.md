Designing Data Augmentation for Simulating Interventions
================================================

by Maximilian Ilse (<ilse.maximilian@gmail.com>), Jakub M. Tomczak and Patrick Forré

Overview
--------

PyTorch implementation of our paper "Designing Data Augmentation for Simulating Interventions":
* Ilse, M., Tomczak, J. M., & Forré, P. (2020). Designing Data Augmentation for Simulating Interventions. Arxiv Link goes here ???

Used modules
------------

- Python 3.6
- PyTorch 1.0.1

Datasets
--------
- MNIST: http://yann.lecun.com/exdb/mnist/
- PACS: https://domaingeneralization.github.io/

Pre-trained AlexNet
-------------------
To reproduce our results on the PACS dataset, please use: https://drive.google.com/file/d/1wUJTH1Joq2KAgrUDeKJghP1Wf7Q9w4z-/view?usp=sharing

Story behind the paper
----------------------
Everybody that works with medical imaging data eventually comes across the following problem: appearance variability. This variability is usually caused by the equipment used to generate medical imaging data, e.g., CT scanners from different vendors will generate images with different intensity patterns. If we train a CNN on data from a single scanner we are likely to overfit on the specific intensity pattern of the scanner. As a result, we are likely to fail to generalize to data from a different scanner.

In late 2018, we started to work on the problem of domain generalization/learning invariant representations motivated by the appearance variability in medical imaging data described above. In domain generalization, one tries to find a representation that generalizes across different environments, called domains, each with a different shift of the input.

This eventually led to a model that we called the Domain Invariant Variational Autoencoder (DIVA, https://arxiv.org/abs/1905.10427, thanks to my co-authors!). While the results of DIVA are promising, there were a couple of experiments that didn’t make it into the paper since the performance of DIVA didn’t match a simple baseline CNN. For a while, we thought it is probably due to optimization issues, etc. During 2019, we realized that we had a very poor understanding of the problem itself.

Citation
--------------------

Please cite our paper if you use this code in your research:
```
???
```

Acknowledgments
--------------------

The work conducted by Maximilian Ilse was funded by the Nederlandse Organisatie voor Wetenschappelijk Onderzoek (Grant DLMedIa: Deep Learning for Medical Image Analysis).
