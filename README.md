# Zero-shot Learning Using Direct Attribute Prediction Method

### Final Project for DL for Spiking Neural Networks and Advanced Data Mining class;

### 01.07.2023, Klagenfurt University S23

Based on the paper "Unseen Object Classes by Between-Class Attribute Transfer"
by C.H. Lampert, H. Nickisch, and S. Harmeling. [1]

## Introduction

The field of object recognition has developed largely over last decades. Training such systems
requires high number of well-labelled training examples. However, data extraction (generation)
and labelling are both tedious as well as financially and time costly procedures. Moreover, in
the vast regions of interest for such models, there could be indefinite number of areas with an
unspecified or lacking data to work on. With help comes ZSL, zero-shot learning technique. "The
core in ZSL is to learn a multi-modal projection between visual and semantic features, by using
labeled seen class data only. Then, the projection is able to generalize well to handle unseen
class data in the test stage." [2] That it to say, it is possible to make predictions about classes
not present during the training phase.

## Datasets description

For this project two datasets will be used, namely AWA2 [3] and CUB-200-2011 [4]. AWA2,
Animals with Attributes 2, is a dataset containing 37322 images of 50 animals with their at-
tributes. For each animal, there are 85 attributes like color, stripe or habitat. Split proposed
by authors will be preserved. The CUB dataset, namely Caltech-UCSD Birds-200-2011, is a
bigger dataset containing 11,788 images of 200 subcategories belonging to birds, each having 312
different attributes. For this dataset, there is also preferred train/test split, and so it will be kept
throughout the experiments. As inputs, instead of images themselves, lower-dimensional (1D)
representation of the image will be used as described in AWA2 paper: "Our image embeddings
are 2048-dim top-layer pooling units of the 101-layered ResNet".[3] The labels are integer values
representing classes. When it comes to attributes, for AWA2 binary attributes are available,
and so 1 means presence of a given attribute of set of 85, 0 - absence. For both AWA2 and
CUB continuous representation of per-class attributes (between 0 and 1) is also available, which
showed [5] to be stronger than binary ones. Training and testing classes are disjoint so that
testing classes have not been seen in the training part.

## Plan

**Goal:** Classify test images into classes which have never been seen
**Method** : Learn attribute classifiers (or regressor) for image representations and then use a
mapping from predicted attributes to class

The idea is to use both binary and continuous representations of attributes and compare
the results between them. The chosen method is DAP - Direct Attribute Prediction [1] which
is traditional attribute-based prediction algorithm. It is an intermediate method, as it requires
an classifier (or a regressor) from images to attributes and only at test time the classes names
can be inferred, so that decision is made based solely on the attribute layer. For CUB dataset,
only continuous attributes are available, so only they will be used. However, for AWA2 both
models involving classification for binary attributes and regression for continuous one will be
build. Supervised classifiers/regressors can be created using different methods, e.g. SVM/SVR
or Neural Nets. Neural Nets approach will be used in this project.

Details avaiable in `Report.pdf`

## Bibliography

[1] C. H. Lampert, H. Nickisch and S. Harmeling, " **Learning to detect unseen object classes
by between-class attribute transfer** ," 2009 IEEE Conference on Computer Vision and Pat-
tern Recognition, Miami, FL, USA, 2009, pp. 951-958, doi: 10.1109/CVPR.2009.5206594.

[2] Y. Liu and T. Tuytelaars, " **A Deep Multi-Modal Explanation Model for Zero-Shot
Learning** ," in IEEE Transactions on Image Processing, vol. 29, pp. 4788-4803, 2020, doi:
10.1109/TIP.2020.2975980.

[3] Xian, Y., Lampert, C. H., Schiele, B., Akata, Z. (2020). **"Zero-Shot Learning – A Com-
prehensive Evaluation of the Good, the Bad and the Ugly"**. arXiv [Cs.CV]. Retrieved
from [http://arxiv.org/abs/1707.](http://arxiv.org/abs/1707.)

[4] Wah, C., Branson, S., Welinder, P., Perona, P., Belongie, S. (2011). , Author = Wah, C.
and Branson, S. and Welinder, P. and Perona, P. and Belongie, S., **CUB dataset** , Year =
2011 Institution = California Institute of Technology, Number = CNS-TR-2011-001. California
Institute of Technology.

[5] Z. Akata, F. Perronnin, Z. Harchaoui and C. Schmid, " **Label-Embedding for Image Clas-
sification** ," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 38, no. 7,
pp. 1425-1438, 1 July 2016, doi: 10.1109/TPAMI.2015.2487986.
