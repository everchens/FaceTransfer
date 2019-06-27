# FaceTransfer
This project borrows code from [MUNIT-Tensorflow](https://github.com/taki0112/MUNIT-Tensorflow) and [StarGAN-Tensorflow](https://github.com/taki0112/StarGAN-Tensorflow).

## Requirements
* Tensorflow 1.4
* Python 3.6

## Usage
```
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.jpg 
           ├── ddd.png
           └── ...


### Train
* python main_2.py --phase train --dataset summer2winter --batch_size 1

### Test
* python main_2.py --phase test --dataset summer2winter --batch_size 1

## Author
everchens
