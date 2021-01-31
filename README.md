# XNOR-Net-Pytorch
You are required to learn how to implemente the gradient calculation of the XNOR-Net[1] (https://github.com/allenai/XNOR-Net). 

[1] Rastegari, M., Ordonez, V., Redmon, J. and Farhadi, A., 2016, October. Xnor-net: Imagenet classification using binary convolutional neural networks. In European conference on computer vision (pp. 525-542). Springer, Cham.

## How to run
To run the training on MNIST using LeNet-5:
```bash
$ cd <Repository Root>/MNIST/
$ python main.py
```
Pretrained model can be downloaded [here](https://drive.google.com/open?id=0B-7I62GOSnZ8R3Jzd0ozdzlJUk0). To evaluate the pretrained model:
```bash
$ cp <Pretrained Model> <Repository Root>/MNIST/models/
$ python main.py --pretrained models/LeNet_5.best.pth.tar --evaluate
```

## Your Tasks
## 1. Read the Notes
### Gradients of scaled sign function
In the paper, the gradient in backward after the scaled sign function is  
  
![equation](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20W_i%7D%3D%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20%7B%5Cwidetilde%7BW%7D%7D_i%7D%20%28%5Cfrac%7B1%7D%7Bn%7D+%5Cfrac%7B%5Cpartial%20sign%28W_i%29%7D%7B%5Cpartial%20W_i%7D%5Ccdot%20%5Calpha%20%29)

<!--
\frac{\partial C}{\partial W_i}=\frac{\partial C}{\partial {\widetilde{W}}_i} (\frac{1}{n}+\frac{\partial sign(W_i)}{\partial W_i}\cdot \alpha )
-->

However, this equation is actually inaccurate. The correct backward gradient should be

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20W_%7Bi%7D%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Ccdot%20sign%28W_%7Bi%7D%29%20%5Ccdot%20%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%5B%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20%5Cwidetilde%7BW%7D_j%7D%20%5Ccdot%20sign%28W_j%29%5D%20&plus;%20%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20%5Cwidetilde%7BW%7D_i%7D%20%5Ccdot%20%5Cfrac%7Bsign%28W_i%29%7D%7BW_i%7D%20%5Ccdot%20%5Calpha)

Details about this correction can be found in the [notes](notes/notes.pdf) (section 1).

## 2. Implemente `updateBinaryGradWeight` function in `util.py`.
Please follow the notes to implement weights gradient calucation. 

## 3. Report results and discussion
***3.1 Please fill the resutls table***
list as tables 

-[ ] Save Memory Footprint
-[ ] Compare With Valina Training
-[ ] Record Running Time

| Dataset  | Network                  | Accuracy                    | Accuracy of floating-point |
|----------|:-------------------------|:----------------------------|:---------------------------|
| MNIST    | ResNet18               |                      |                   |
|----------|:-------------------------|:----------------------------|:---------------------------|
| Cifar10  | ResNet18                |                      |                   |
***3.2 Hyper-parameters***

Please show how the variations on number of channels and filter size will affect speedup.
Please refer to Fig.4(b-c) in the original paper.

***3.3 Discuss your results***
Submit a reports (specified)
