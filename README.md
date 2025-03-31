# MMR-Mamba
MMR-Mamba: Multi-Modal MRI Reconstruction with Mamba and Spatial-Frequency Information Fusion

Pytorch Code for the paper ["MMR-Mamba: Multi-Modal MRI Reconstruction with Mamba and Spatial-Frequency Information Fusion"](https://arxiv.org/abs/2406.18950)


# Environment

CUDA Version: 11.7

python=3.8.18 

pytorch=1.13.1

# Inroduction

Multi-modal MRI offers valuable complementary information for diagnosis and treatment; however, its utility is limited by prolonged scanning times. To accelerate the acquisition process, a practical approach is to reconstruct images of the target modality, which requires longer scanning times, from under-sampled k-space data using the fully-sampled reference modality with shorter scanning times as guidance. The primary challenge of this task is comprehensively and efficiently integrating complementary information from different modalities to achieve high-quality reconstruction. Existing methods struggle with this: 1) convolution-based models fail to capture long-range dependencies; 2) transformer-based models, while excelling in global feature modeling, struggle with quadratic computational complexity. To address this, we propose MMR-Mamba, a novel framework that thoroughly and efficiently integrates multi-modal features for MRI reconstruction, leveraging Mamba's capability to capture long-range dependencies with linear computational complexity while exploiting global properties of the Fourier domain. Specifically, we first design a Target modality-guided Cross Mamba (TCM) module in the spatial domain, which maximally restores the target modality information by selectively incorporating relevant information from the reference modality. Then, we introduce a Selective Frequency Fusion (SFF) module to efficiently integrate global information in the Fourier domain and recover high-frequency signals for the reconstruction of structural details. Furthermore, we devise an Adaptive Spatial-Frequency Fusion (ASFF) module, which mutually enhances the spatial and frequency domains by supplementing less informative channels from one domain with corresponding channels from the other.


## Links for downloading the public datasets:

1) BraTS20 Dataset - <a href="https://www.kaggle.com/datasets/awsaf49/brats2020-training-data"> Link </a> 
2) fastMRI Dataset - <a href="https://fastmri.med.nyu.edu/"> Link </a>

We provide the IDs of the data used in the BraTS dataset as CSV files, located in the ./dataloaders/cv_splits_100patients/ folder. These files contain the patient IDs corresponding to the dataset splits.

## Train MMR-Mamba
```bash 
bash train_munet.sh
```

## Ackonwledgements

We give acknowledgements to [fastMRI](https://github.com/facebookresearch/fastMRI), [MTrans
](https://github.com/chunmeifeng/MTrans), and [Pan-Mamba](https://github.com/alexhe101/Pan-Mamba).


# Citation
```bash
@article{zou2025mmr,
  title={MMR-Mamba: Multi-modal MRI reconstruction with Mamba and spatial-frequency information fusion},
  author={Zou, Jing and Liu, Lanqing and Chen, Qi and Wang, Shujun and Hu, Zhanli and Xing, Xiaohan and Qin, Jing},
  journal={Medical Image Analysis},
  pages={103549},
  year={2025},
  publisher={Elsevier}
}
```
