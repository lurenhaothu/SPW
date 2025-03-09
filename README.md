# SPW loss
The PyTorch implementation of the CWMI loss proposed in paper: **Steerable Pyramid Weighted Loss: Multi-Scale Adaptive Weighting for Semantic Segmentation**. <br>

---


## Abstract
Semantic segmentation is a core task in computer vision with applications in biomedical imaging, remote sensing, and autonomous driving. While standard loss functions such as cross-entropy and Dice loss perform well in general cases, they often struggle with fine structures, particularly in tasks involving thin structures or closely packed objects. Various weight map-based loss functions have been proposed to address this issue by assigning higher loss weights to pixels prone to misclassification. However, these methods typically rely on precomputed or runtime-generated weight maps based on distance transforms, which impose significant computational costs and fail to adapt to evolving network predictions. In this paper, we propose a novel steerable pyramid-based weighted (SPW) loss function that efficiently generates adaptive weight maps. Unlike traditional boundary-aware losses that depend on static or iteratively updated distance maps, our method leverages steerable pyramids to dynamically emphasize regions across multiple frequency bands (capturing features at different scales) while maintaining computational efficiency. Additionally, by incorporating network predictions into the weight computation, our approach enables adaptive refinement during training. We evaluate our method on the SNEMI3D, GlaS, and DRIVE datasets, benchmarking it against 11 state-of-the-art loss functions. Our results demonstrate that the proposed SPW loss function achieves superior pixel precision and segmentation accuracy with minimal computational overhead. This work provides an effective and efficient solution for improving semantic segmentation, particularly for applications requiring multiscale feature representation. 

<p align = "center">
<img src="figures/Figure 1.PNG">
</p>

---


## **Environment**
Ensure you have the following dependencies installed:

```sh
Python 3.12.8
PyTorch 2.5.1+cu121
```

Additional dependencies are listed in [`requirements.txt`](./requirements.txt). Install them using:

```sh
pip install -r requirements.txt
```

---

## **Dataset Preparation**
Download the datasets and place them in the following directories:

| Dataset       | Download Link | Expected Directory |
|--------------|--------------|--------------------|
| **SNEMI3D**  | [Zenodo](https://zenodo.org/record/7142003) | `./data/snemi3d/` |
| **GlaS**     | [Kaggle](https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation) | `./data/GlaS/Warwick_QU_Dataset/` |
| **DRIVE**    | [Kaggle](https://www.kaggle.com/datasets/yurekanramasamy/drive-dataset) | `./data/DRIVE/Drive_source/` |

---

## **Usage**
### **1. Prepare Datasets**
To preprocess datasets and calculate the mean and standard deviation for each dataset, run:

```sh
python ./data/dataprepare.py
```

### **2. Train the Model**
Run the training script:

```sh
run train.ipynb
```

### **3. Evaluate the Model**
Run the evaluation script:

```sh
run eval.ipynb
```

---

## **Additional Weight Map-Based Loss Functions**
SPW loss does **not** require additional data preparation. However, if you wish to test other weight map-based loss functions, run the corresponding scripts:

- **U-Net weighted cross entropy (WCE)** ([arXiv:1505.04597](https://arxiv.org/abs/1505.04597))
  ```sh
  python ./data/map_gen_unet.py
  ```
- **ABW loss** ([arXiv:1905.09226v2](https://arxiv.org/abs/1905.09226v2))
  ```sh
  python ./data/map_gen_ABW.py
  ```
- **Skea_topo loss** ([arXiv:2404.18539](https://arxiv.org/abs/2404.18539))
  ```sh
  python ./data/skeleton_aware_loss_gen.py
  python ./data/skeleton_gen.py
  ```

---


## Results
<p align = "center">
<img src="figures/Table 1.PNG">
</p>

### SNEMI3D
<p align = "center">
<img src="figures/Figure 2.PNG">
</p>

### GlaS
<p align = "center">
<img src="figures/Figure 3.PNG">
</p>

### DRIVE
<p align = "center">
<img src="figures/Figure 4.PNG">
</p>


### Computational cost
<p align = "center">
<img src="figures/Table 3.PNG">
</p>
