# Machine Learning Engineer Nanodegree

Marc Seitz
March 2020

## Proposal

### Domain Background
Convolutional neural networks (CNNs) is a technique that analyses visual imagery with the help of multiple (deep) neural layers. It can be used to classify images, recognize objects, or detect anomalies in speech waveforms. In any case, the input should be an image in some way. In this project, we will be looking at image classification, in particular, classification of dog breeds. On one hand, this classification sounds simple (especially to a dog lover), however, there are several breeds that are closely related and the differences are subtle, difficult even for the human eye. Since the beginning of the ImageNet competition, CNNs have been become better than humans at classification problems. Now, CNNs are applied to more and more narrow fields to see if the network can find even the most benign features in an image.

### Problem Statement
Image classification means returning a single class for an input image. In contrast, object recognition, which sounds similar, returns a list of all object classes found in the image. In our case, we have a list of classes of dogs and upon an input of a random dog/human/non-dog image, we would like to return the correct class of dog breed. In the case of human image, we would like to return a class of dog breed lookalike. In the case of non-dog/non-human image we should return an error.

### Dateset and Inputs
We are using two separate datasets for [dogs](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) and [humans](http://vis-www.cs.umass.edu/lfw/lfw.tgz) provided by Udacity and The University of Massachusetts, respectively.

The dog dataset contains 133 folders each corresponding to a different dog breed. 

### Solution Statement
The aim is to create a CNN-based classifier that can correctly classify each of the 133 dog breeds. In order to classify the dog breeds properly, I am designing a deep neural network and then training the network on the supplied datasets. In the end, the classifier can differentiate between dog breeds as well as dogs and humans and non-human/non-dogs.

We will use the evaluation metrics to compare the the performance of the classifier against the benchmark models in the next section.

### Benchmark Model
For the benchmark model, we will use models that have performed well on a similar dataset of dog breeds from Stanford [1](http://cs231n.stanford.edu/reports/2015/pdfs/fcdh_FinalReport.pdf)

| Method                         | Top - 1 Accuracy (%) |
|--------------------------------|----------------------|
| SIFT + Gaussian Kernel         | 22%                  |
| Unsupervised Learning Template | 38%                  |
| Gnostic Fields                 | 47%                  |
| Selective Pooling Vectors      | 52%                  |