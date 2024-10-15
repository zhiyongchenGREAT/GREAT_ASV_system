# GREAT Speaker Recognition

A comprehensive implementation of advanced embedding learning techniques for text-independent speaker verification, based on the following papers:

1. **A Study on Angular Based Embedding Learning for Text-independent Speaker Verification**
2. **Triplet Based Embedding Distance and Similarity Learning for Text-independent Speaker Verification**
3. **Supervised Imbalanced Multi-domain Adaptation for Text-independent Speaker Verification**

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

This repository provides implementations of cutting-edge techniques in speaker verification, focusing on:

- **Angular-based Embedding Learning**: Enhancing the discriminative power of speaker embeddings using angular margin-based loss functions.
- **Triplet-based Embedding Distance and Similarity Learning**: Employing triplet loss strategies to maximize inter-speaker variance and minimize intra-speaker variance.
- **Supervised Imbalanced Multi-domain Adaptation**: Addressing domain adaptation challenges with supervised methods for imbalanced multi-domain data.

## Features

- Implementation of angular margin-based loss functions (e.g., AM-Softmax, AAM-Softmax).
- Triplet loss with various mining strategies (hard, semi-hard, soft).
- Supervised domain adaptation techniques for imbalanced multi-domain datasets.
- Support for popular speaker verification datasets:
  - **VoxCeleb**
  - **CN-Celeb**
  - **DeepMine**
- Modular and extensible codebase for easy experimentation.
- Pre-trained models and configuration files for reproducing results.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.7 or higher
- CUDA 10.1 or higher (for GPU support)

## Usage
### Dataset
- Voxceleb 1 and Voxceleb 2
- CNCeleb 1 and CNCeleb 2
- Deepmine
### Training
- Code recipe in train_dist/
### Evaluations
- Evaluation recipe in score/
## Citation
If you find this repository useful in your research, please cite the following papers:

-Chen, Zhiyong, Zongze Ren, and Shugong Xu. "A study on angular based embedding learning for text-independent speaker verification." 2019 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC). IEEE, 2019.

-Ren, Zongze, Zhiyong Chen, and Shugong Xu. "Triplet based embedding distance and similarity learning for text-independent speaker verification." 2019 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC). IEEE, 2019.

-Chen, Zhiyong, Zongze Ren, and Shugong Xu. "Supervised Imbalanced Multi-domain Adaptation for Text-independent Speaker Verification." Proceedings of the 2020 9th International Conference on Computing and Pattern Recognition. 2020.

## License
This project is licensed under the MIT License.

## Acknowledgments
We thank the authors of the papers for their contributions to the field of speaker recognition. We also appreciate the open-source community for their valuable tools and resources.


