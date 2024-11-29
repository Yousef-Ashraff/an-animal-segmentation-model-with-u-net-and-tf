# Animal Segmentation Dataset 
* Dataset Link: [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

* Deployment Link:

* The dataset consists of several images  and masks divided into the following 2 classes:
  - Cats
  - Dogs
* The dataset consists of 7349 images and masks
* In this notebook uses U-Net model
* U-Net architecture which I implement but with input image size (128,128,3)
![U-Net architecture](images/u-net-architecture.png)

* U-Net model with
  - training accuracy 90.7%
  - validation accuracy 90.5%
  - test accuracy 90.6%

*The model runed 20 epoch and returned the best model in epoch 10 with augmentation

*Some sample of images
![Cat image and mask](images/Cat_1.png)
![Dog image and mask](images/Dog_1.png)


 
