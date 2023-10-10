[![Python CI](https://github.com/hussainsan/tensorflow_cifar/actions/workflows/python_ci.yml/badge.svg)](https://github.com/hussainsan/tensorflow_cifar/actions/workflows/python_ci.yml)

# 🚀 Train CIFAR10/Cifar100 with Tensorflow 2

This repository contains implementations of various neural network architectures trained on the [CIFAR10 and CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) datasets using TensorFlow 2. The CIFAR10 and CIFAR100 datasets consist of 60,000 32x32 color images in 10 and 100 classes, respectively.


## 📂 Directory Structure
```plaintext
.
├── LICENSE
├── Makefile
├── README.md
├── data_utils.py       # Utilities for data processing
├── main.py             # Main training and testing script
├── nn_models           # Neural network model implementations
│   ├── mobilenetv2.py
│   ├── resnet.py
│   └── ........ 
└── requirements.txt    # Required libraries and dependencies
```

## 🛠️ Installation:
```
make all
```
### or use pip to install:
```
pip install -r requirements.txt
```

## 🤖 Run models on CIFAR10 using available models (e.g. MobileNetV2):
```
    python main.py train --dataset=cifar10 --epochs=5 --batch_size=128 --model=MobileNetV2

```

## 🤖 Similarily run models on CIFAR100 using available models (e.g. Resnet18):
```
    python main.py train --dataset=cifar10 --epochs=100 --batch_size=128 --model=ResNet18

```
## 📋 To list all the available models:
```
    python main.py train --list_models

```

## 🧪 To test the model after training (note the model will be save automatically after training):
```
    python main.py test --dataset=cifar10 --model=MobileNetV2

```


## 📊 Currently Training and Testing the models. Results will be published soon. 

| Model           |Parameters (M)|CIFAR10 Test Acc.|
|-----------------|---------|---------|
| ResNet18        |11184650| 92.36%	  |
| ResNet34        |21311754| 92.39%	  |
| ResNet50        |23547402| 92.04%	  |
| ResNet101       |42565642| 91.52%	  |
| ResNet152       |58232330| 91.30%	  |
| MobileNetV2     |2952362| ------  |
| DenseNet121     |7043658| 91.86%	  |
| VGG16           |33638218| 91.86%	  |

## 📢 Feedback & Contribution
Feedback is always welcome! If you have suggestions or want to contribute to this repository, please create an issue or a pull request.
