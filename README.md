# MADAN

A Pytorch Code for [Multi-source Domain Adaptation for Semantic Segmentation](https://arxiv.org/abs/1910.12181)

If you use this code in your research please consider citing:

```
@InProceedings{zhao2019madan,
   title = {Multi-source Domain Adaptation for Semantic Segmentation},
   author = {Zhao, Sicheng and Li, Bo and Yue, Xiangyu and Gu, Yang and Xu, Pengfei and Tan, Hu, Runbo and Chai, Hua and   Keutzer, Kurt},
   booktitle = {Advances in Neural Information Processing Systems},
   year = {2019}
}
```

## Quick Look

Our multi-source domain adaptation builds on the work [CyCADA](https://github.com/jhoffman/cycada_release) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Since we focus on Semantic Segmentation task, we remove Digit Classfication part in CyCADA.

We add following modules and achieve startling improvements.

1. Dynamic Semantic Consistency Module
2. Adversarial Aggregation Module
   1. Sub-domain Aggregation Discriminator
   2. Cross-domain Cycle Discriminator

While we implements [MDAN](https://openreview.net/pdf?id=ryDNZZZAW) for Semantic Segmentation task in Pytorch as our baseline comparasion.

## Overall Structure

![image-20190608104531451](http://ww4.sinaimg.cn/large/006tNc79ly1g3tjype7qlj31vo0u0hb1.jpg)

## Setup

Check out this repo:

```bash
git clone https://github.com/pikachusocute/MADAN.git
```

Install Python3 requirements

```bash
pip3 install -r requirements.txt
```

## Dynamic Adversarial Image Generation

We follow the way in CyCADA, in the first step, we need to train Image Adaptation module to transfer source image(GTA, Synthia or Multi-source) to "source as target".

![image-20190608111738818](http://ww4.sinaimg.cn/large/006tNc79ly1g3tkvxw9rrj31r40e8kjl.jpg)

We refer Image Adaptation module from GTA to Cityscapes as GTA->Cityscapes in the following.

#### GTA->Cityscapes

```bash
cd scripts/CycleGAN
bash cyclegan_gta2cityscapes.sh
```

In the training process, snapshot files will be stored in `cyclegan/checkpoints/[EXP_NAME]`.

Usually, afer we run for 20 epochs, there'll be a file `20_net_G_A.pth ` in previous folder path. 

Then we run the test process.

```bash
bash scripts/CycleGAN/test_templates.sh [EXP_NAME] 20 cycle_gan_semantic_fcn gta5_cityscapes
```

In multi-source case, there are both `20_net_G_A_1.pth` and `20_net_G_A_2.pth` exist. We use another script to run test process.

![image](https://tva1.sinaimg.cn/large/006y8mN6ly1g9cqt9m2kmj31j80skgsh.jpg)

```bash
bash scripts/CycleGAN/test_templates_cycle.sh [EXP_NAME] 20 test synthia_cityscapes gta5_cityscapes
```

New dataset will be generated at `~/cyclegan/results/[EXP_NAME]/train_20`.

After we obtain a new source stylized dataset, we then train segmenter on the new dataset.

## Pixel Level Adaptation

In this part, we train our new segmenter on new dataset.

```bash
ln -s ~/cyclegan/results/[EXP_NAME]/train_20 ~/data/cyclegta5/[EXP_NAME]_TRAIN_60
```

Then we set `dataflag = [EXP_NAME]_TRAIN_60` to find datasets' paths, and follow instructions to train segmenter to perform pixel level adaptation.

```bash
bash scripts/FCN/train_fcn8s_cyclesgta5_DSC.sh
```

## Feature Level Adaptation

For adaptation, we use

```bash
bash scripts/ADDA/adda_cyclegta2cs_score.sh
```

Make sure you choose the desired `src` and `tgt` and `datadir` before. In this process, you should load your `base_model` trained on synthetic dataset and perform adaptation in feature level to real scene dataset.

### Our Model

We release our adaptation model in the `./models`, you can use `scripts/eval_templates.sh` to evaluate its validity.

1. [CycleGTA5_Dynamic_Semantic_Consistency](https://drive.google.com/file/d/1moGF7L2hkTHUPUzqsSxPwKNlHCHQm4Ms/view?usp=sharing)
2. [CycleSYNTHIA_Dynamic_Semantic_Consistency](https://drive.google.com/file/d/19V5J1zyF3ct3247gSSr-u3WVkDJqQvUk/view?usp=sharing)
3. [Multi_Source_SAD_CCD](https://drive.google.com/file/d/1xgmLwhsbwv-isy7R5FkNevVSH9gcMxuq/view?usp=sharing)

### Transfered Dataset

We will release our transfer dataset soon, where our `CycleGTA5_Dynamic_Semantic_Consistency` model is trained to perform pixel level adaptation.
