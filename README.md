# Surgical Tool Localization

Surgical Tool Classification, Detection, Localization to assess surgical performance and efficiency

## Purpose of ML App

## Outline

## Software Dependencies

- Nvidia DALI Pipeline
- TensorFlow-GPU 2.7.3 or PyTorch 1.1
- Unity 2021 LTS

## Setup Software Dev Environment

## How to Run Demo

## ML Pipeline Approaches

### Approach 1

**Methodology**

From research paper [1A], they designed a lightweight attention-guided CNN that inherits the advantages of the single- and two-stage detection methods and works more accurately and efficiently than RefineDet.

Their proposed approach performed the STD via a coarse detection module (CDM) and a refined detection module (RDM). Their method achieved end-to-end training by using multi-task loss function. Their distance interfsection-over-union non-maximum suppression (DIoU-NMS) was proposed to post-process the tools detection results. Refer to the figure 1 diagram in the paper [3A]. Their CDM is powered by a modified VGG-16 and it does binary classification to decide whether the anchor is a tool or background filtering out a large number of negative anchors (cases where there isnt a tool in the frame). RDM consists of multiple conv layers and SENet to generate accurate locations and classification scors of surgical tools.

### Approach 2

For this custom approach, there are 2 DNNs that come from research paper [2A].

1. Heatmap Hourglass CNN Network output the heatmaps which represent the location of the instruments tip area

2. Bounding-box regression network is a modified version of VGG-16 originally used for image classification.

They compared their model against Faster RCNN, YOLOv3 and Retinanet with a Non-maximum suppression (NMS) with a threshold of 0.5

Their DNN Networks: mAP1 = 91.60 %, mAP2 = 100.00 %, Detection Time (per frame)s = 0.023

YOLOv3 (Darknet-53): mAP1 = 90.92 %, mAP2 = 99.07 %, Detection time (per frame)s = 0.034

- Refer to network diagrams in research paper [2A].

### Approach 3

1. Train a surgical tool detector on image-level labels and learn the whole region boundaries of the surgical tools in laparoscopic videos.

2. Employ a Convolutional LSTM (ConvLSTM) to learn the spatio-temporal coherence across the surgical video frames for tool localization and tracking.

**Methodology**

From research paper [3A], there models are built on **ResNet-18**.

Their **Detector** is their own custom FCN Baseline with ResNet-18, 7-channel Lh-map with spatial pooling outputting 7 classes for 7 surgical tools.

Their **Tracker** is their own custom ConvLSTM and it leverages the separation of the tool type in the 7-channel Lh-map from the FCN Detector to build a baseline model for tool tracking.

**IMPORTANT**: Their **ConvLSTM** tracker model trained on videos at **1fps** can generalize to unlabelled videos at **25fps**, potentially making it unconstrained by the **fps**.

- Refer to network diagrams in research paper [3A].

### Approach 4

For this custom approach, there are 2 DNNs that come from research paper [4A].

- First a Faster RCNN based on VGG-16 network takes a laparascopic surgical video. Then Region Proposal Network (RPN) shares convolutional feattures with object detection networks. For each input image, the RPN generates region proposals to contain an object and features pooled over these regions before being passed to a final classification and bounding box refinement network. The output is spatial bounding box positions of detected surgical tools in the video frame.

- Refer to network diagrams in research paper [4A].


## Dataset Links

- Download [Surgical Tool Localization in Endoscopic Videos 109GB Dataset](https://surgtoolloc.grand-challenge.org/data-download/) of 24K Video Clips from Endoscopic Vision Challenge 2022
    - [Surgical Tool Endoscopic Video Dataset Description](https://surgtoolloc.grand-challenge.org/data/)


- Download [2016 M2CAI Tool Presence Detection Challenge](https://ai.stanford.edu/~syyeung/tooldetection.html) which includes a m2cai16-tool dataset
    - m2cai16-tool consists of 15 videos total recorded at 25 fps of cholecystectomy procedures performed at the University Hospital of Strasbourg France where each video is labeled with binary annotations indicating tool presence
    - From m2cai-tool dataset, first 10 videos are used for training RCNN model and 11-15 videos are used for testing that model
    - To the best of Stanford's team knowledge, there was no such dataset that currently exists for real-world laparosopic surgical videos. Thus, from the m2cai16-tool dataset, they created m2cai16-tool-locations with spatial annotations of the tools.
    - For their m2cai16-tool dataset, they label 2532 of the frames under supervision and spot-checking from a surgeon, with the coordinates of spatial bounding boxes around the tools. They usee 50%, 30% and 20% for training, validation and test spliits. The 2532 frames were selected from among the 23,000 total frames. 
        - Again these 23,000 frames were from videos whose durations range from 20 to 75 minutes downsampled to 1 fps for processing and labeled with binary annotations indicating presence or absence of seven surgical tools: grasper, bipolar, hook, scissors, clip applier, irrigator and specimen bag.

- Download [JIGSAWS: The JHU-ISI Gesture and Skill Assessment Working Set](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/). This dataset is used with Deep Learning Models for Evaluating, Identifying Surgical Actions and Measuring Performance as noted in research paper [6A].
    - More info on the dataset can be found in research paper [7A].

## Research Publication Links

### AI/DL Perspective

- [1A] P. Shi, Z. Zhao, S. Hu and F. Chang, "**Real-Time Surgical Tool Detection in Minimally Invasive Surgery Based on Attention-Guided Convolutional Neural Network**," in IEEE Access, vol. 8, pp. 228853-228862, 2020, doi: 10.1109/ACCESS.2020.3046258.: https://ieeexplore.ieee.org/document/9301279

- [2A] Zhao Z, Cai T, Chang F, Cheng X. **Real-time surgical instrument detection in robot-assisted surgery using a convolutional neural network cascade.** Healthc Technol Lett. 2019 Nov 26;6(6):275-279. doi: 10.1049/htl.2019.0064. PMID: 32038871; PMCID: PMC6952255.: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6952255/

- [3A] Chinedu Innocent Nwoye, Didier Mutter, Jacques Marescaux, Nicolas Padoy. **Weakly Supervised Convolutional LSTM Approach for Tool Tracking in Laparoscopic Video.** International Journal of Computer Assisted Radiology and Surgery, Springer Verlag, 2019: https://arxiv.org/pdf/1812.01366v2.pdf
    - YouTube Demo Video 1: [Weakly Supervised Convolutional LSTM Approach for Tool Tracking in Laparoscopic Videos](https://www.youtube.com/watch?v=vnMwlS5tvHE)
    - YouTube Demo Video 2: [Weakly Supervised Convolutional LSTM Approach for Tool Tracking (Test at 25 FPS)](https://www.youtube.com/watch?v=SNhd1yzOe50)

- [4A] A. Jin, et al., "**Tool Detection and Operative Skill Assessment in Surgical Videos Using Region-Based Convolutional Neural Networks**," in 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), Lake Tahoe, NV, USA, 2018 pp. 691-699.
doi: 10.1109/WACV.2018.00081
keywords: {tools;videos;task analysis;training;minimally invasive surgery;convolutional neural networks}
url: https://doi.ieeecomputersociety.org/10.1109/WACV.2018.00081

- [5A] Roß, Tobias, and Lena Maier-Hein. "**Surgical Data Science in Endoscopic Surgery.**" German Cancer Research Center (DKFZ), Department of Computer Assisted Medical Interventions, 2020. 
    - https://archiv.ub.uni-heidelberg.de/volltextserver/30928/1/PhD_Thesis_Tobias_Ross.pdf

- [6A] Khalid S, Goldenberg M, Grantcharov T, Taati B, Rudzicz F. **Evaluation of Deep Learning Models for Identifying Surgical Actions and Measuring Performance.** JAMA Netw Open. 2020;3(3):e201664. doi:10.1001/jamanetworkopen.2020.1664
    - https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2763474

- [7A] Yixin Gao, S. Swaroop Vedula, Carol E. Reiley, Narges Ahmidi, Balakrishnan Varadarajan, Henry C. Lin, Lingling Tao, Luca Zappella, Benjam ́ın B ́ejar, David D. Yuh, Chi Chiung Grace Chen, Ren ́e Vidal, Sanjeev Khudanpur and Gregory D. Hager, **The JHU-ISI Gesture and Skill Assessment Working Set (JIGSAWS): A Surgical Activity Dataset for Human Motion Modeling,** In Modeling and Monitoring of Computer Assisted Interventions (M2CAI) – MICCAI Workshop, 2014
    - https://cirl.lcsr.jhu.edu/wp-content/uploads/2015/11/JIGSAWS.pdf

### UI/UX Perspective

- [1B] Gasques, D., Johnson, J.G., Sharkey, T., Feng, Y., Wang, R., Xu, Z.R., Zavala, E., Zhang, Y., Xie, W., Zhang, X., Davis, K.L., Yip, M.C., & Weibel, N. (2021). **ARTEMIS: A Collaborative Mixed-Reality System for Immersive Surgical Telementoring.** Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems.
    - https://dl.acm.org/doi/fullHtml/10.1145/3411764.3445576
