<h1 align="center"> SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery 
 </a> <a href=""><img src="https://img.shields.io/badge/CVPR-2024-blue"></a> </h1>
<p align="center">
<h4 align="center">The official repo for [CVPR'24] <a href="https://openaccess.thecvf.com/content/CVPR2024/html/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.html">SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery </a>.</h4>
<h5 align="center"><em>Xin Guo*, Jiangwei Lao*, Bo Dang*, Yingying Zhang, Lei Yu, Lixiang Ru, Liheng Zhong, Ziyuan Huang, Kang Wu, Dingxiang Hu, Huimei He, Jian Wang, Jingdong Chen, Ming Yang &dagger;, Yongjun Zhang, Yansheng Li &dagger;</em></h5>
<h6 align="center">* Equally contributing first authors.  &dagger; Corresponding authors.</h6>
<p align="center">
  <a href="#news">Updates</a> |
  <a href="#abstract">Abstract</a> |
  <a href="#method">Method</a> |
  <a href="#installation">Installation</a> |
  <a href="#usage">Usage</a> |
  <a href="#license">License</a> |
  <a href="#citation">Citation</a>
</p>

# Updates

- **2025.08.04**: Our latest work, [SkySense++](https://www.nature.com/articles/s42256-025-01078-8), has been accepted by Nature Machine Intelligence. The pretrained weights and code are available at [this repository](https://github.com/kang-wu/SkySensePlusPlus).
- **2024.06.17**: SkySense has been accepted to CVPR2024. The pretrained weight is available at [this repository](https://www.notion.so/SkySense-Checkpoints-a7fcff6ce29a4647a08c7fe416910509).
- **2023.12.01**: A collection of papers, datasets, benchmarks, code, and pretrained weights for Remote Sensing Foundation Models (RSFMs) is available [here](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models).

# Abstract
<div align="justify">
Prior studies on Remote Sensing Foundation Model (RSFM) reveal immense potential towards a generic model for Earth Observation. Nevertheless, these works primarily focus on a single modality without temporal and geo-context modeling, hampering their capabilities for diverse tasks. In this study, we present SkySense, a generic billion-scale model, pre-trained on a curated multi-modal Remote Sensing Imagery (RSI) dataset with 21.5 million temporal sequences. SkySense incorporates a factorized multi-modal spatiotemporal encoder taking temporal sequences of optical and Synthetic Aperture Radar (SAR) data as input. This encoder is pre-trained by our proposed Multi-Granularity Contrastive Learning to learn representations across different modal and spatial granularities. To further enhance the RSI representations by the geo-context clue, we introduce Geo-Context Prototype Learning to learn region-aware prototypes upon RSI's multi-modal spatiotemporal features. To our best knowledge, SkySense is the largest Multi-Modal RSFM to date, whose modules can be flexibly combined or used individually to accommodate various tasks. It demonstrates remarkable generalization capabilities on a thorough evaluation encompassing 16 datasets over 7 tasks, from single- to multi-modal, static to temporal, and classification to localization. SkySense surpasses 18 recent RSFMs in all test scenarios. Specifically, it outperforms the latest models such as GFM, SatLas and Scale-MAE by a large margin, i.e., 2.76%, 3.67% and 3.61% on average respectively. We will release the pre-trained weights to facilitate future research and Earth Observation applications.
</div>

# Method
<p align="center">
<img src="assets/skysense.png" width="700">
</p>

SkySense, **a multi-modal remote sensing foundation model (MM-RSFM)**, features a modular design capable of handling diverse tasks ranging from single- to multi-modal, static to temporal, and classification to localization. The design incorporates three novel technical components: a) A factorized multi-modal spatiotemporal encoder to effectively process multi-modal temporal remote sensing imagery; b) Multi-Granularity Contrastive Learning that learns features at various levels of granularities to facilitate different tasks; and c) Geo-Context Prototype Learning to extract region-aware geo-context clue to enable implicit geo-knowledge integration. Extensive comparisons with 18 recently published RSFMs reveal that SkySense achieves the state-of-the-art performance. 

&#128522; We hope the release of pre-trained weights will contribute to the Remote Sensing community and facilitate future research. &#128640;&#128640;&#128640;


# Installation

## HuggingFace Transformers (Recommended)

Requires only **PyTorch 2.0+** and **HuggingFace Transformers**. No mmcv, mmcls, mmseg, or mmdet required.

Step 1. Create a conda environment and activate it.

```bash
conda create --name skysense python=3.10
conda activate skysense
```

Step 2. Install PyTorch (2.0+) and HuggingFace Transformers.
```bash
pip install torch>=2.0 transformers
```

## Legacy Installation (mmcv-based, for downstream detection/segmentation configs)

<details>
<summary>Click to expand legacy installation</summary>

Step 1. Create a conda environment and activate it. Install gdal 3.4.0.

```bash
conda create --name skysense python=3.8
conda activate skysense
conda install -c conda-forge gdal=3.4.0
```

Step 2. Install PyTorch following official instructions. Pytorch 1.13.1 is recommend.
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 torchtext==0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

Step 3. Install MMCV 1.7.1, MMDetection 2.28.2, MMCls 0.25.0 and MMSegmentation 0.30.0 using MIM.
```bash
pip install -U openmim
mim install mmcv-full==1.7.1
pip install mmcls==0.25.0 mmsegmentation==0.30.0 mmdet==2.28.2 yapf==0.40.1 timm==0.6.13 rasterio==1.2.10 scikit-learn==1.2.2
```

</details>

# Usage

The following describes how to utilize SkySense's pluggable components to adapt to high-resolution RGB imagery, Sentinel-2 multispectral imagery, Sentinel-1 SAR imagery, and so on.

## Convert Checkpoint to HuggingFace Format

Convert original SkySense pre-trained weights to HuggingFace format:

```bash
# Convert HR/RGB backbone (Swin Transformer V2 - Huge)
python scripts/convert_checkpoint_to_hf.py \
    --input-path skysense_pretrained.pth \
    --data-type rgb \
    --output-dir ./skysense-swinv2-huge-rgb

# Convert Sentinel-2 backbone (Vision Transformer - Large)
python scripts/convert_checkpoint_to_hf.py \
    --input-path skysense_pretrained.pth \
    --data-type s2 \
    --output-dir ./skysense-vit-large-s2

# Convert Sentinel-1 backbone (Vision Transformer - Large)
python scripts/convert_checkpoint_to_hf.py \
    --input-path skysense_pretrained.pth \
    --data-type s1 \
    --output-dir ./skysense-vit-large-s1

# Convert RGBNIR backbone (Swin Transformer V2 - Huge, 4 channels)
python scripts/convert_checkpoint_to_hf.py \
    --input-path skysense_pretrained.pth \
    --data-type rgbnir \
    --output-dir ./skysense-swinv2-huge-rgbnir
```

## Load Pre-trained Models (HuggingFace)

```python
# For high-resolution RGB imagery (Swin Transformer V2 - Huge)
from skysense import SkySenseSwinV2Model
model = SkySenseSwinV2Model.from_pretrained('./skysense-swinv2-huge-rgb')
```

```python
# For Sentinel-2 imagery (Vision Transformer - Large)
from skysense import SkySenseViTModel
model = SkySenseViTModel.from_pretrained('./skysense-vit-large-s2')
```

## Feature Extraction Pipeline

```python
import torch
from skysense import SkySenseSwinV2Model, SkySenseSwinV2Config
from skysense.pipeline import SkySenseFeatureExtractionPipeline

# Initialize model (from pretrained or fresh)
config = SkySenseSwinV2Config(arch='huge', in_channels=3)
model = SkySenseSwinV2Model(config)

# Create pipeline
pipe = SkySenseFeatureExtractionPipeline(model=model, device='cpu')

# Extract features
features = pipe(torch.randn(1, 3, 224, 224))
print(features['last_hidden_state'].shape)
```

## Initialize Models from Scratch

```python
from skysense import SkySenseSwinV2Config, SkySenseSwinV2Model
from skysense import SkySenseViTConfig, SkySenseViTModel

# Swin Transformer V2 - Huge for RGB (default config)
config = SkySenseSwinV2Config(arch='huge', in_channels=3)
model = SkySenseSwinV2Model(config)

# Vision Transformer - Large for Sentinel-2 (10-band, default config)
config = SkySenseViTConfig(in_channels=10)
model = SkySenseViTModel(config)

# Vision Transformer - Large for Sentinel-1 (2-band)
config = SkySenseViTConfig(in_channels=2)
model = SkySenseViTModel(config)
```

## Load Pre-trained Weights (Legacy)

<details>
<summary>Click to expand legacy usage</summary>

```python
# For high-resolution RGB imagery (band order: R, G, B) or RGBNIR imagery (band order: R, G, B, NIR). 
# Model architecture: Swin Tranformer v2 - Huge
import torch
from models.swin_transformer_v2 import SwinTransformerV2

checkpoint = torch.load('skysense_model_backbone_hr.pth')
checkpoint = {k.replace('backbone.', ''): v for k, v in checkpoint.items() if k.startswith('backbone.')}
swinv2_model = SwinTransformerV2()
msg = swinv2_model.load_state_dict(checkpoint, strict=False)
# missing_keys=['stages.0.blocks.0.attn.w_msa.relative_coords_table', ...], unexpected_keys=['mask_token']
swinv2_model = swinv2_model.cuda()
```
```python
# For Sentinel-2 imagery (band order: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12) or Sentinel-1 imagery (band order: VV, VH). 
# Model architecture: Vision Transformer - Large
import torch
from models.vision_transformer import VisionTransformer

checkpoint = torch.load('skysense_model_backbone_s2.pth')
vit_model = VisionTransformer()
msg = vit_model.load_state_dict(checkpoint, strict=False)
# missing_keys=[], unexpected_keys=['ctpe']
vit_model = vit_model.cuda()
```

</details>

## Example Usage of Downstream Tasks & Results

Note: All results were obtained using NVIDIA A100 GPUs (80GB).

### Semantic Segmentation

|    Dataset     |  Metric  | Performance (%) | Config/Code |
|:---------------------------|:-------:|:-------:|:------:|
|  iSAID                     |   mIoU  |   70.91 |  [Config](SkySense/segmentation/configs/swin_transformer_v2/upernet_swinv2_huge_patch4_window8_896x896_80k_isaid.py)      |

### Object Detection

|    Dataset     |  Metric  | Performance (%) | Config/Code |
|:---------------------------|:-------:|:-------:|:------:|
|  DIOR                      |   mAP50  |   78.73 | [Config](SkySense/detection/configs/swin_transformer_v2/faster_rcnn_swinv2_huge_patch4_window8_fpn-1x_dior.py)       |

# License
The pre-trained model weights are only available for the non-commercial research. For any commercial use or cooperation, please contact Yansheng Li at Wuhan University (e-mail: yansheng.li@whu.edu.cn).

# Citation
If you find our repo useful, please consider giving a star and citation:

```
@InProceedings{Guo_2024_CVPR,
    author    = {Guo, Xin and Lao, Jiangwei and Dang, Bo and Zhang, Yingying and Yu, Lei and Ru, Lixiang and Zhong, Liheng and Huang, Ziyuan and Wu, Kang and Hu, Dingxiang and He, Huimei and Wang, Jian and Chen, Jingdong and Yang, Ming and Zhang, Yongjun and Li, Yansheng},
    title     = {SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {27672-27683}
}

@article{wu2025semantic,
  author       = {Wu, Kang and Zhang, Yingying and Ru, Lixiang and Dang, Bo and Lao, Jiangwei and Yu, Lei and Luo, Junwei and Zhu, Zifan and Sun, Yue and Zhang, Jiahao and Zhu, Qi and Wang, Jian and Yang, Ming and Chen, Jingdong and Zhang, Yongjun and Li, Yansheng},
  title        = {A semantic‑enhanced multi‑modal remote sensing foundation model for Earth observation},
  journal      = {Nature Machine Intelligence},
  year         = {2025},
  doi          = {10.1038/s42256-025-01078-8},
  url          = {https://doi.org/10.1038/s42256-025-01078-8}
}

@inproceedings{zhu2025skysense,
  title={Skysense-o: Towards open-world remote sensing interpretation with vision-centric visual-language modeling},
  author={Zhu, Qi and Lao, Jiangwei and Ji, Deyi and Luo, Junwei and Wu, Kang and Zhang, Yingying and Ru, Lixiang and Wang, Jian and Chen, Jingdong and Yang, Ming and others},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={14733--14744},
  year={2025}
}

@article{luo2024skysensegpt,
  title={Skysensegpt: A fine-grained instruction tuning dataset and model for remote sensing vision-language understanding},
  author={Luo, Junwei and Pang, Zhen and Zhang, Yongjun and Wang, Tingzhu and Wang, Linlin and Dang, Bo and Lao, Jiangwei and Wang, Jian and Chen, Jingdong and Tan, Yihua and others},
  journal={arXiv preprint arXiv:2406.10100},
  year={2024}
}
```
For any other questions please contact bodang@whu.edu.cn.




