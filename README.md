# Back-propagation (Summer Course homework 1)
It's only using **Numpy** packages to two-layer neural network.  

You can get some detailed introduction and experimental results in the link below.  
https://github.com/secondlevel/Back-propagation/blob/main/Experiment%20Report.pdf

<p float="center">
  <img src="https://github.com/secondlevel/Back-propagation/blob/main/Experiment%20Result/%E6%9E%B6%E6%A7%8B%E5%9C%96.PNG" title="Architecture" hspace="150" />
</p>

## Hardware
Operating System: Windows 10  

CPU Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz  

GPU 0 NVIDIA GeForce GTX TITAN X  

## Requirement

In this work, I use Anaconda to manage my environment.

```bash=
$ conda create --name backwardenv python=3.8
$ conda install numpy
$ conda install matplotlib -y 
$ conda install pandas -y
```

## Data description


## Training and Testing the Linear classifier Result.

```python=
$ python linear_training_testing.py
```

You will get the following linear classifier result.

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
  <img src="Experiment Result/linear_classifier_training_curve.PNG" width="450" title="training curve" hspace="0" />
  <img src="Experiment Result/linear_classifier_testing_result.PNG" width="450" title="testing result" hspace="20" />
</p>


## Training and Testing the XOR classifier Result.

```python=
$ python XOR_training_testing.py
```


