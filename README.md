#P2PNet (ICCV2021 Oral Presentation)
This repository contains codes for the official implementation in PyTorch of P2PNet as described in Rethinking Counting and Localization in Crowds: A Purely Point-Based Framework.

A brief introduction of P2PNet can be found at 机器之心 (almosthuman).

The codes is tested with PyTorch 1.5.0. It may not run with other versions.

Visualized demos for P2PNet

![image](https://github.com/user-attachments/assets/8372d6e0-53e3-452e-b9d6-2d9d725d64fc)

![image](https://github.com/user-attachments/assets/a06f6974-79a1-4f9b-9465-555c4380d424)

![image](https://github.com/user-attachments/assets/c5d0c808-e456-4efe-b64e-b0982e649bd3)


The network

The overall architecture of the P2PNet. Built upon the VGG16, it firstly introduce an upsampling path to obtain fine-grained feature map. Then it exploits two branches to simultaneously predict a set of point proposals and their confidence scores.

![image](https://github.com/user-attachments/assets/16604700-008a-4dc1-b023-9e9447f37506)


Comparison with state-of-the-art methods

The P2PNet achieved state-of-the-art performance on several challenging datasets with various densities.

![image](https://github.com/user-attachments/assets/144070fb-f462-497b-ab2c-87991d3f8db0)

Comparison on the NWPU-Crowd dataset.

![image](https://github.com/user-attachments/assets/d56e7eaa-ad7c-4221-a4eb-e01bd62aa3c8)


The overall performance for both counting and localization.

![image](https://github.com/user-attachments/assets/3d327446-f26d-4574-b378-f78c7f8ddd67)

Comparison for the localization performance in terms of F1-Measure on NWPU.

![image](https://github.com/user-attachments/assets/9323babb-5888-4bf6-aa55-74fb639accc6)

Installation

Clone this repo into a directory named P2PNET_ROOT

Organize your datasets as required

Install Python dependencies. We use python 3.6.5 and pytorch 1.5.0

pip install -r requirements.txt

Organize the counting dataset

We use a list file to collect all the images and their ground truth annotations in a counting dataset. When your dataset is organized as recommended in the following, the format of this list file is defined as:


train/scene01/img01.jpg train/scene01/img01.txt

train/scene01/img02.jpg train/scene01/img02.txt

...

train/scene02/img01.jpg train/scene02/img01.txt

Dataset structures:


DATA_ROOT/
        |->train/
        |    |->scene01 /
        |    |->scene02/
        |    |->...
        |->test/
        |    |->scene01/
        |    |->scene02/
        |    |->...
        |->train.list
        |->test.list
        
DATA_ROOT is your path containing the counting datasets.


Annotations format

For the annotations of each image, we use a single txt file which contains one annotation per line. Note that indexing for pixel values starts at 0. The expected format of each line is:


x1 y1
x2 y2
...


Training
The network can be trained using the train.py script. For training on SHTechPartA, use


CUDA_VISIBLE_DEVICES=0 python train.py --data_root $DATA_ROOT \
    --dataset_file SHHA \
    --epochs 3500 \
    --lr_drop 3500 \
    --output_dir ./logs \
    --checkpoints_dir ./weights \
    --tensorboard_dir ./logs \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --eval_freq 1 \
    --gpu_id 0

By default, a periodic evaluation will be conducted on the validation set.


Testing


A trained model (with an MAE of 51.96) on SHTechPartA is available at "./weights", run the following commands to launch a visualization demo:



CUDA_VISIBLE_DEVICES=0 python run_test.py --weight_path ./weights/SHTechA.pth --output_dir ./logs/

Acknowledgements

Part of codes are borrowed from the C^3 Framework.

We refer to DETR to implement our matching strategy.

Citing P2PNet

If you find P2PNet is useful in your project, please consider citing us:


@inproceedings{song2021rethinking,
  title={Rethinking Counting and Localization in Crowds: A Purely Point-Based Framework},
  author={Song, Qingyu and Wang, Changan and Jiang, Zhengkai and Wang, Yabiao and Tai, Ying and Wang, Chengjie and Li, Jilin and Huang, Feiyue and Wu, Yang},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}


Related works from Tencent Youtu Lab

[AAAI2021] To Choose or to Fuse? Scale Selection for Crowd Counting. (paper link & codes)
[ICCV2021] Uniformity in Heterogeneity: Diving Deep into Count Interval Partition for Crowd Counting. (paper link & codes)
