# Back-propagation (Deep Learning and Practice homework 1)
It's only using **Numpy** packages to build two-layer neural network.  

You can get some detailed introduction and experimental results in the link below.  
https://github.com/secondlevel/Back-propagation/blob/main/Experiment%20Report.pdf

<p float="center">
  <img src="https://github.com/secondlevel/Back-propagation/blob/main/Experiment%20Result/%E6%9E%B6%E6%A7%8B%E5%9C%96.PNG" title="Architecture" width="500" />
</p>

## Hardware
Operating System: Windows 10  

CPU: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz  

GPU: NVIDIA GeForce GTX TITAN X  

## Requirement

In this work, you can use the following two option to build the environment.

### First option
```bash
$ conda env create -f environment.yml
```

### Second option
```bash
$ conda create --name Summer python=3.8 -y
$ conda activate Summer
$ conda install numpy
$ conda install matplotlib -y 
$ conda install pandas -y
```

## System Architecture

<p float="center">
  <img src="Experiment Result/System_architecture.PNG" title="Architecture" width="600" />
</p>

## Data Description

In this project, the training data and testing data can be generate by **generate_data.py** file.  

This file contains two function.

- #### generator_linear function

  - input: linear classifier coordinate (x,y) 
  - output: data coordinate, data label(blue point and red point)
  - commend: 
  ```python=
  input, label = generator_linear(1000)
  ```

- #### generator_XOR_easy function
  - input: XOR classifier coordinate (x,y) 
  - output: data coordinate, data label(blue point and red point) 
  - commend:
  ```python=
  input, label = generator_XOR_easy(11)
  ```

## Performance Metrics

In this project, **Accuracy** is the classification metrics, and **Mean Squared Error** is the loss function. 

### Accuracy

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/151693558-d2ea220b-607b-41c3-9d03-e01d0682aaed.gif" title="Accuracy" width="300" />
</p>    

- **True Positive(TP)** signifies how many positive class samples your model predicted correctly.
- **True Negative(TN)** signifies how many negative class samples your model predicted correctly.
- **False Positive(FP)** signifies how many negative class samples your model predicted incorrectly. This factor represents Type-I error in statistical nomenclature. This error positioning in the confusion matrix depends on the choice of the null hypothesis.
- **False Negative(FN)** signifies how many positive class samples your model predicted incorrectly. This factor represents Type-II error in statistical nomenclature. This error positioning in the confusion matrix also depends on the choice of the null hypothesis. 

---

### Mean Squared Error(MSE)

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/151695030-61f1f71e-6fb6-498d-bee4-ef85a0b5d959.gif" title="Mean Squared Error(MSE)" width="200" />
</p>  

- **y_j:** ground-truth value
- **y_hat:** predicted value from the regression model
- **N:** number of datums

## Training and Testing the Linear classifier Result.

**You can see the detailed experiment result in the [link](https://github.com/secondlevel/Back-propagation/blob/main/Experiment%20Report.pdf).**  

The first, you need to config the training parameters through **linear_training_testing.py** file as following.

```bash=
236     data_number = 1000
237
238     TwoLayerNet = [
239          {"input_dim": 2, "output_dim": 20, "activation": "sigmoid"},
240          {"input_dim": 20, "output_dim": 10, "activation": "sigmoid"},
241          {"input_dim": 10, "output_dim": 1, "activation": "sigmoid"},
242      ]
```

Then you can use the following instruction to training and testing the linear classifier.  

```python=
$ python linear_training_testing.py
```

You will get the following linear classifier training and testing result.  

```bash=
Epoch: 005910  Loss: 0.014239  Accuracy: 0.999000
Epoch: 005920  Loss: 0.014215  Accuracy: 0.999000
Epoch: 005930  Loss: 0.014192  Accuracy: 0.999000
Epoch: 005940  Loss: 0.014168  Accuracy: 0.999000
Epoch: 005950  Loss: 0.014145  Accuracy: 0.999000
Epoch: 005960  Loss: 0.014122  Accuracy: 0.999000
Epoch: 005970  Loss: 0.014099  Accuracy: 0.999000
Epoch: 005980  Loss: 0.014075  Accuracy: 0.999000
Epoch: 005990  Loss: 0.014052  Accuracy: 0.999000
``` 

<p float="center">
  <img src="Experiment Result/linear_classifier_training_curve.PNG" width="460" title="training curve" hspace="0" />
  <img src="Experiment Result/linear_classifier_testing_result.PNG" width="460" title="testing result" hspace="20" />
</p>


## Training and Testing the XOR classifier Result.

**Detailed experiment result link: https://github.com/secondlevel/Back-propagation/blob/main/Experiment%20Report.pdf**  

The first, you need to config the training parameters through **XOR_training_testing.py** file.   

```bash=
237     data_number = 22
238
239     TwoLayerNet = [
240          {"input_dim": 2, "output_dim": 20, "activation": "sigmoid"},
241          {"input_dim": 20, "output_dim": 10, "activation": "sigmoid"},
242          {"input_dim": 10, "output_dim": 1, "activation": "sigmoid"},
243      ]
```

Then you can use the following instruction to training and testing the XOR classifier.  

```python=
$ python XOR_training_testing.py
```

You will get the following XOR classifier training and testing result.  

```bash=
Epoch: 005880  Loss: 0.005006  Accuracy: 1.000000
Epoch: 005890  Loss: 0.004990  Accuracy: 1.000000
Epoch: 005900  Loss: 0.004975  Accuracy: 1.000000
Epoch: 005910  Loss: 0.004960  Accuracy: 1.000000
Epoch: 005920  Loss: 0.004945  Accuracy: 1.000000
Epoch: 005930  Loss: 0.004929  Accuracy: 1.000000
Epoch: 005940  Loss: 0.004915  Accuracy: 1.000000
Epoch: 005950  Loss: 0.004900  Accuracy: 1.000000
Epoch: 005960  Loss: 0.004885  Accuracy: 1.000000
Epoch: 005970  Loss: 0.004870  Accuracy: 1.000000
Epoch: 005980  Loss: 0.004855  Accuracy: 1.000000
Epoch: 005990  Loss: 0.004841  Accuracy: 1.000000
``` 

<p float="center">
  <img src="Experiment Result/XOR_classifier_training_curve.PNG" width="460" title="training curve" hspace="0" />
  <img src="Experiment Result/XOR_classifier_testing_result.PNG" width="460" title="testing result" hspace="20" />
</p>

##  Reference
- https://zhuanlan.zhihu.com/p/89391305  
- https://www.itread01.com/content/1546354994.html  
- https://www.itread01.com/feffx.html  
- https://zhuanlan.zhihu.com/p/55497753  
- https://www.gushiciku.cn/pl/2Jeg/zh-tw  
- https://kknews.cc/zh-tw/news/ze2voea.html    
