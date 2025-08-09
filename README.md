CrackVision: Effective Concrete Crack Detection With Deep Learning and Transfer Learning
This repository contains the official implementation for the paper "CrackVision: Effective Concrete Crack Detection With Deep Learning and Transfer Learning". The project provides a comprehensive analysis of deep learning models for automated crack detection and classification in concrete structures.

Overview
The manual inspection of concrete structures is a time-consuming, subjective, and often hazardous process. This project, CrackVision, presents a deep learning approach to automate this task, improving efficiency and accuracy. We compare three state-of-the-art pre-trained convolutional neural network (CNN) architectures—ResNet50, Xception, and InceptionV3—on two distinct classification tasks using public datasets. The results demonstrate the high effectiveness of transfer learning for this purpose, with the Xception model achieving a peak accuracy of 99.97%.

Publication
The work in this repository is detailed in the following paper:

Title: CrackVision: Effective Concrete Crack Detection With Deep Learning and Transfer Learning

Authors: Abdulrahman A. Alkannad, Ahmad Al Smadi, Shuyuan Yang, Mutasem K. Al-Smadi, Moeen Al-Makhlafi, Zhixi Feng, and Zhenlong Yin.

PDF: A copy of the paper, CrackVision_Effective_Concrete_Crack_Detection_With_Deep_Learning_and_Transfer_Learning.pdf, is included in this repository.

[Link to Online Publication] (<- Replace this with the public link to your paper if available, e.g., on arXiv, ResearchGate, or a conference website)

Models and Results
This project is organized into three main classification challenges, with each task implemented using three different model architectures. The highest accuracy achieved for each binary classification task is noted below.

Classification Task

Dataset

Model Architecture

Top Accuracy (Test Set)

Notebook File

Binary (Crack vs. No-Crack)

METU

ResNet50

99.95%

Crackclassification RESENT50 with METU model.ipynb

Binary (Crack vs. No-Crack)

METU

Xception

99.97%

Crackclassification Xception with METU model.ipynb

Binary (Crack vs. No-Crack)

METU

InceptionV3

99.96%

Crackclassification InceptionV3 with METU model.ipynb

Multi-Class (Deck vs. Wall vs. Pavement)

SDNET2018

ResNet50

99.90%

Crack classification RESENT50 D VS W VS P.ipynb

Multi-Class (Deck vs. Wall vs. Pavement)

SDNET2018

Xception

99.92%

Crack classification Xception D VS W VS P.ipynb

Multi-Class (Deck vs. Wall vs. Pavement)

SDNET2018

InceptionV3

99.91%

Crack classification InceptionV3 D VS W VS P.ipynb

Multi-Class (Deck vs. Pavement)

SDNET2018

ResNet50

99.88%

Crack classification RESENT50 D VS P.ipynb

Multi-Class (Deck vs. Pavement)

SDNET2018

Xception

99.95%

Crack classification Xception D VS P.ipynb

Multi-Class (Deck vs. Pavement)

SDNET2018

InceptionV3

99.93%

Crack classification InceptionV3 D VS P.ipynb

Setup and Installation
To reproduce the results, follow these steps to set up the environment.

1. Prerequisites
Python 3.8+

Pip (Python package installer)

Git version control

(Recommended) An NVIDIA GPU with CUDA and cuDNN for accelerated training.

2. Clone the Repository
Open your terminal or command prompt and clone this repository:

git clone https://github.com/ALKANNAD/CrackVision.git
cd CrackVision

3. Set Up a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

# Create the virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

4. Install Dependencies
Install all required Python libraries using the requirements.txt file.

pip install -r requirements.txt

5. Download the Datasets
You only need to download the dataset for the specific experiment you wish to run.

For the METU Dataset (Binary Classification):

The dataset contains 40,000 images (20,000 cracked, 20,000 non-cracked).

Download from Kaggle.

Place the Positive and Negative image folders inside a METU directory at the project root.

For the SDNET2018 Dataset (Multi-Class Classification):

The dataset contains over 56,000 images of bridge decks, walls, and pavements.

Download from Mendeley Data.

Place the unzipped image folders (e.g., CD, CW, CP, UD, etc.) inside a Dataset/SDNET2018 directory at the project root.

Usage
To run an experiment:

Launch Jupyter Notebook or Jupyter Lab from your terminal:

jupyter lab

In the browser window, navigate to and open the desired .ipynb notebook from the table above.

Run the notebook cells sequentially. The notebooks will preprocess the data, build the model, train it, and save the evaluation results.

Repository Structure
CrackVision/
├── METU/                   # For the METU dataset (needs to be created)
│   ├── Negative/
│   └── Positive/
├── Dataset/                # For the SDNET2018 dataset (needs to be created)
│   └── SDNET2018/
│       ├── CD/
│       ├── CP/
│       ├── CW/
│       └── ...
├── Models/                   # Directory where trained models and plots are saved
├── CrackVision_Effective_Concrete_Crack_Detection_With_Deep_Learning_and_Transfer_Learning.pdf
├── *.ipynb                   # All Jupyter Notebooks for the experiments
└── README.md

Citation
If you use this code or the findings from our paper in your research, please cite our work.

@article{CrackVision2025,
    title={CrackVision: Effective Concrete Crack Detection With Deep Learning and Transfer Learning},
    author={Alkannad, Abdulrahman A. and Al Smadi, Ahmad and Yang, Shuyuan and Al-Smadi, Mutasem K. and Al-Makhlafi, Moeen and Feng, Zhixi and Yin, Zhenlong},
    year={2025},
    journal={Journal/IEEE Access},
}

(Please update the year and journal/conference details as appropriate)
