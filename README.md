## GULF: GUided Learning through successive Functional gradient optimization

This repository provides the author implementation of GULF, described in [[Johnson & Zhang, ICML2020]](http://riejohnson.com/paper/gulf-icml20.pdf).  It also provides the author implementation (in pyTorch) of deep pyramid convolutional neural networks ([DPCNN](description/dpcnn.md)) [[Johnson & Zhang ACL2017]](http://riejohnson.com/paper/dpcnn-acl17.pdf) used in the GULF experiments on text classification.  

**_Requirements_**

* Python version 3
* pyTorch 1.2.0 and torchvision 0.4.0 (or higher)
* pip install -r requirements.txt

**_Examples_**

* To perform GULF2 with ini:random with ResNet-28 on CIFAR100: `python3 train_cifar.py`
* To change the initialization to 'ini:base': `python3 train_cifar.py --ini_type iniBase`
* To write test error etc. to a file in the csv format: `python3 train_cifar.py --csv_fn results.csv`
* To get help: `python3 train_cifar.py -h`

Similarly, `train_svhn.py`, `train_imgnet.py`, and `train_yelp` can be used.  
`train_yelp.py` for sentiment classification and [DPCNN](description/dpcnn.md) are explained [here](description/dpcnn.md). 

**_Example configurations_**

code         | CPU cores     | CPU memory | GPU
------------ | ------------- | ---------- | ---
train_cifar.py | 3  | 20GB | 1
train_svhn.py  | 2  | 20GB | 1
train_imgnet.py, resnet50   | 8 | 128GB | 2
train_imgnet.py, wrn50-2    | 12 | 128GB | 4
train_yelp.py  | 1  | 24GB | 1

GPU device memory: 12GB

**_Notes_**

* The code uses a GPU whenever it is available.  To avoid use of GPUs even when it is available, 
  empty `CUDA_VISIBLE_DEVICES` via shell before calling python.  
  
        export CUDA_VISIBLE_DEVICES=""
* The code writes a lot to stdout, and so it is recommended to redirect it to a file.  

**_References_**

[[Johnson & Zhang, 2020]](http://riejohnson.com/paper/gulf-icml20.pdf) Guided Learning of Nonconvex Models through Successive Functional Gradient Optimization.  Rie Johnson and Tong Zhang.  ICML-2020.   
[[Johnson & Zhang, 2017]](http://riejohnson.com/paper/dpcnn-acl17.pdf) Deep pyramid convolutional neural networks for text categorization.  Rie Johnson and Tong Zhang.  ACL-2017.   
