# MobileHand: Real-time 3D Hand Shape and Pose Estimation from Color Image

This repository contains the sample code for the paper MobileHand: Real-time 3D Hand Shape and Pose Estimation from Color Image

[**Paper**]() | [**Video**]() | [**Results on STB Dataset B1 Random**](https://www.youtube.com/embed/vIfJSPhprnU) | [**Results on STB Dataset B1 Counting**](https://www.youtube.com/embed/qrmRm-gF02A)

If you find our code or paper useful, please consider citing
```
@inproceedings{MobileHand:2020,
  title = {MobileHand: Real-time 3D Hand Shape and Pose Estimation from Color Image},
  author = {Guan Ming, Lim and Prayook, Jatesiktat and Wei Tech, Ang},
  booktitle = {27th International Conference on Neural Information Processing (ICONIP)},
  year = {2020}
}
```

## Installation
The simplest way to run our implementation is to use anaconda.

You can create an anaconda environment called `mobilehand` with
```
conda env create -f environment.yaml
conda activate mobilehand
```

Next, you will need to download the MANO right hand model
* Go to [MANO project page](http://mano.is.tue.mpg.de/)
* Click on _Sign In_ and register for your account
* Download Models & Code (`mano_v1_2.zip`)
* Unzip and copy the file `mano_v1_2/models/MANO_RIGHT.pkl` into the `mobilehand/model` folder

To allow the use of MANO model in Python 3 environment, we will need to remove Chumpy objects from the original `MANO_RIGHT.pkl` model. The following steps are adapted from [smplx repo](https://github.com/vchoutas/smplx/blob/master/tools/README.md):
* In a Python 2 virtual environment with [Chumpy](https://github.com/mattloper/chumpy) and [tqdm](https://github.com/tqdm/tqdm) installed
```
conda create -n py27 python=2.7
conda activate py27
pip install chumpy
pip install tqdm
```

* Run the following command to remove any Chumpy objects and it will create a new file `MANO_RIGHT_NEW.pkl`:
```
python model/clean_ch.py --input-models model/MANO_RIGHT.pkl --output-folder model/
```

## Demo
Change directory to the folder `mobilehand/code/`
```
cd code/
```
To test on a sample image from the STB dataset run:
```
python demo.py --dataset stb
```
To test on a sample image from the FreiHAND dataset run:
```
python demo.py --dataset freihand
```

## Sample results

- STB dataset\
![](data/stb_result.png)

- FreiHAND dataset\
![](data/freihand_result.png)


## References for dataset

##### [2017 ICIP] A Hand Pose Tracking Benchmark from Stereo Matching. [\[PDF\]](https://ieeexplore.ieee.org/document/8296428)  [\[Project\]](https://sites.google.com/site/zhjw1988/) [\[Code\]](https://github.com/zhjwustc/icip17_stereo_hand_pose_dataset)
*Jiawei Zhang, Jianbo Jiao, Mingliang Chen, Liangqiong Qu, Xiaobin Xu, and Qingxiong Yang*


##### [ICCV 2019] FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images. [\[PDF\]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zimmermann_FreiHAND_A_Dataset_for_Markerless_Capture_of_Hand_Pose_and_ICCV_2019_paper.pdf) [\[Project\]](https://lmb.informatik.uni-freiburg.de/projects/freihand/) [\[Code\]](https://github.com/lmb-freiburg/freihand)
_Christian Zimmermann, Duygu Ceylan, Jimei Yang, Bryan Russell, Max Argus, Thomas Brox_


## References on 3D hand shape and pose estimation from color image

##### [CVPR 2019] Pushing the Envelope for RGB-based Dense 3D Hand Pose Estimation via Neural Rendering. [\[PDF\]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Baek_Pushing_the_Envelope_for_RGB-Based_Dense_3D_Hand_Pose_Estimation_CVPR_2019_paper.pdf)
_Seungryul Baek, Kwang In Kim, Tae-Kyun Kim_


##### [CVPR 2019] 3D Hand Shape and Pose from Images in the Wild. [\[PDF\]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Boukhayma_3D_Hand_Shape_and_Pose_From_Images_in_the_Wild_CVPR_2019_paper.pdf) [\[Code\]](https://github.com/boukhayma/3dhand)
_Adnane Boukhayma, Rodrigo de Bem, Philip H.S. Torr_


##### [CVPR 2019] 3D Hand Shape and Pose Estimation from a Single RGB Image. [\[PDF\]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ge_3D_Hand_Shape_and_Pose_Estimation_From_a_Single_RGB_CVPR_2019_paper.pdf) [\[Project\]](https://sites.google.com/site/geliuhaontu/home/cvpr2019) [\[Code\]](https://github.com/3d-hand-shape/hand-graph-cnn) *(Oral)*
_Liuhao Ge, Zhou Ren, Yuncheng Li, Zehao Xue, Yingying Wang, Jianfei Cai, Junsong Yuan_


##### [CVPR 2019] Learning joint reconstruction of hands and manipulated objects. [\[PDF\]](https://arxiv.org/pdf/1904.05767.pdf) [\[Code\]](https://github.com/hassony2/manopth) [\[Code\]](https://github.com/hassony2/obman_train) [\[Project\]](https://www.di.ens.fr/willow/research/obman/)
_Yana Hasson, Gül Varol, Dimitris Tzionas, Igor Kalevatykh, Michael J. Black, Ivan Laptev, and Cordelia Schmid_


##### [ICCV 2019] End-to-end Hand Mesh Recovery from a Monocular RGB Image. [\[PDF\]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_End-to-End_Hand_Mesh_Recovery_From_a_Monocular_RGB_Image_ICCV_2019_paper.pdf)  [\[Code\]](https://github.com/Wavelet303/HAMR)
_Xiong Zhang\*, Qiang Li\*, Wenbo Zhang, Wen Zheng_


##### [CVPR 2020] Weakly-Supervised Mesh-Convolutional Hand Reconstruction in the Wild. [\[PDF\]](https://arxiv.org/pdf/2004.01946.pdf) [\[Project\]](https://www.arielai.com/mesh_hands/)  *(Oral)*
_Dominik Kulon, Riza Alp Güler, Iasonas Kokkinos, Michael Bronstein, Stefanos Zafeiriou_


##### [CVPR 2020] Monocular Real-time Hand Shape and Motion Capture using Multi-modal Data. [\[PDF\]](https://arxiv.org/pdf/2003.09572.pdf) [\[Project\]](https://calciferzh.github.io/publications/zhou2020monocular) [\[Code\]](https://github.com/CalciferZh/minimal-hand)
_Yuxiao Zhou, Marc Habermann, Weipeng Xu, Ikhsanul Habibie, Christian Theobalt, Feng Xu_


## References on other key methods that influence this work

##### [MVA 2019] Accurate Hand Keypoint Localization on Mobile Devices. [\[PDF\]](http://users.ics.forth.gr/~argyros/mypapers/2019_05_MVA_hand2Dkeypoints.pdf) [\[Code\]](https://github.com/FORTH-ModelBasedTracker/MonocularRGB_2D_Handjoints_MVA19)
_Filippos Gouidis, Paschalis Panteleris, Iason Oikonomidis, Antonis Argyros_


##### [CVPR 2018] End-to-end Recovery of Human Shape and Pose. [\[PDF\]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Kanazawa_End-to-End_Recovery_of_CVPR_2018_paper.pdf) [\[Project\]](https://akanazawa.github.io/hmr/) [\[Code\]](https://github.com/akanazawa/hmr)
_Angjoo Kanazawa, Michael J Black, David W. Jacobs, Jitendra Malik_


##### [SIGGRAPH ASIA 2017] Embodied Hands:Modeling and Capturing Hands and Bodies Together. [\[PDF\]](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/392/Embodied_Hands_SiggraphAsia2017.pdf) [\[Project\]](https://mano.is.tue.mpg.de/)
_Javier Romero, Dimitrios Tzionas, Michael J Black_
