# FaceTransfer
This project borrows code from [MUNIT-Tensorflow](https://github.com/taki0112/MUNIT-Tensorflow) and [StarGAN-Tensorflow](https://github.com/taki0112/StarGAN-Tensorflow).

![A2B_1](https://github.com/everchens/FaceTransfer/blob/master/A_187_01_style0.png)![A2B_2](https://github.com/everchens/FaceTransfer/blob/master/A_1_01_style0.png)

![B2A_1](https://github.com/everchens/FaceTransfer/blob/master/B_102_01_style0.png)![B2A_2](https://github.com/everchens/FaceTransfer/blob/master/B_169_01_style0.png)

[Gakki to Trump](https://youtu.be/y30jtjCZA64)

[Trump to Gakki](https://youtu.be/IhM9F-wWv7U)

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
