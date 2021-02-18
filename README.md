# Binarized Neural Networks: XNOR-Net
You are required to learn how to implemente the gradient calculation of the XNOR-Net[1] (https://github.com/allenai/XNOR-Net). 

[1] Rastegari, M., Ordonez, V., Redmon, J. and Farhadi, A., 2016, October. Xnor-net: Imagenet classification using binary convolutional neural networks. In European conference on computer vision (pp. 525-542). Springer, Cham.

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

You will implement the `updateBinaryGradWeight` for both `MNIST/util.py` and `CIFAR_10/util.py`. 

## 3. Report results and discussion
### 3.1 Please fill the resutls table

You need to evaluate the efficiency and accuracy of binary convolution vs. standard convolution (floating-point). 

You will perform the evaluation on MNIST and Cifar10 datasets. You will evaluate LeNet5 for MNIST and ResNet18 for Cifar10. MNIST experiment is in `MNIST` folder. Cifar10 experiment is in `CIFAR_10` folder. Here I show a example about how to run MNIST with LeNet5.

***How to run***

To run the training on MNIST using LeNet-5:
```bash
$ cd <Repository Root>/MNIST/
$ python main.py
```
You can validate your result with the pretrained model. Pretrained model can be downloaded [here](https://drive.google.com/open?id=0B-7I62GOSnZ8R3Jzd0ozdzlJUk0). To evaluate the pretrained model:
```bash
$ cp <Pretrained Model> <Repository Root>/MNIST/models/
$ python main.py --pretrained models/LeNet_5.best.pth.tar --evaluate
```

***Compare Model Accuracy***

| Dataset  | Network                  | Vanilla Net (floating-point)                   | XNOR-NET |
|----------|:-------------------------|:----------------------------|:---------------------------|
| MNIST    | LeNet5              |                      |                   |
| Cifar10  | ResNet18                |                      |                   |

***Compare Memory Comsumption for the Trained Model***
| Dataset  | Network                  | Vanilla Net (floating-point)                   | XNOR-NET |
|----------|:-------------------------|:----------------------------|:---------------------------|
| MNIST    | LeNet5              |                      |                   |
| Cifar10  | ResNet18                |                      |                   |


***Compare Model Training Time***
| Dataset  | Network                  | Vanilla Net (floating-point)                   | XNOR-NET |
|----------|:-------------------------|:----------------------------|:---------------------------|
| MNIST    | LeNet5              |                      |                   |
| Cifar10  | ResNet18                |                      |                   |


***Compare Model Inference Time***
| Dataset  | Network                  | Vanilla Net (floating-point)                   | XNOR-NET |
|----------|:-------------------------|:----------------------------|:---------------------------|
| MNIST    | LeNet5              |                      |                   |
| Cifar10  | ResNet18                |                      |                   |

### 3.2 Hyper-parameters

Please show how the variations on number of channels and filter size will affect speedup.
Please refer to Fig.4(b-c) in the original paper.

### 3.3 Discuss your results
Please describe the settings of your experiments. Please include the required results (described in 3.1 and 3.2). Please add captions to describe your figures and tables. Please analyze the advangetages and limitations of XNOR-NET. It would be best to write brief discussions on your results, such as the patterns (what and why), conclusions, and any observations you want to discuss. 


