# Machine Learning Engineer Nanodegree

Marc Seitz
March 2020

## Proposal

### Domain Background
Convolutional neural networks (CNNs) is a technique that analyses visual imagery with the help of multiple (deep) neural layers. It can be used to classify images, recognize objects, or detect anomalies in speech waveforms. In any case, the input should be an image in some way. In this project, we will be looking at image classification, in particular, classification of dog breeds. On one hand, this classification sounds simple (especially to a dog lover), however, there are several breeds that are closely related and the differences are subtle, difficult even for the human eye. Since the beginning of the ImageNet competition, CNNs have been become better than humans at classification problems. Now, CNNs are applied to more and more narrow fields to see if the network can find even the most benign features in an image.

### Problem Statement
Image classification means returning a single class for an input image. In contrast, object recognition, which sounds similar, returns a list of all object classes found in the image. In our case, we have a list of classes of dogs and upon an input of a random dog/human/non-dog image, we would like to return the correct class of dog breed. In the case of human image, we would like to return a class of dog breed lookalike. In the case of non-dog/non-human image we should return an error.