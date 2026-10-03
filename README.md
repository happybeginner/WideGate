# WideGate: Beyond Directed Acyclic Graph Learning in Subcircuit Boundary Prediction


Official code repository for the paper: WideGate: Beyond Directed Acyclic Graph Learning in Subcircuit Boundary Prediction

Predicting subcircuit boundaries is crucial for several EDA tasks in logical analysis and design. This paper identifies two fundamental shortcomings in existing GNN models when handling this task: the difficulty in considering boundary intercorrelation and neighbor heterophily, which severely limits the models’ performance. To address these challenges, we propose a novel GNN model, WideGate, incorporating a receptive field generation module and an adaptive aggregation module. If you plan to explore more potential innovation points, please feel free to discuss with us (Email: liu_jiawei@bupt.edu.cn). We are looking forward to collaborate with you!

## Abstract
 Abstract—Subcircuit boundary prediction is an important application of machine learning in logical analysis, effectively supporting tasks such as functional verification and logic optimization. Existing methods often convert circuits into and inverter graphs and then use directed acyclic graph neural networks to perform this task. However, two key characteristics of subcircuit boundary prediction do not align with the fundamental assumptions of directed acyclic graph (DAG) learning, which limits the model’s expressiveness and generalization capabilities. To break these assumptions, we propose WideGate, which includes a receptive field generation module that extends beyond the fanin cone and fanout cone, as well as an adaptive aggregation module that focuses on boundaries. Extensive experiments show that WideGate significantly outperforms existing methods in terms of prediction accuracy and training efficiency for subcircuit boundary prediction. 

## Installation
```bash
conda create -n widegate python=3.9.18
conda activate widegate
pip install -r requirements.txt
```

## Directory Structure
```
WideGate
  ├── FGNN_relate
  ├── kernel
  ├── results
  ├── __init__.py
  ├── batch.py
  ├── dataloader.py
  ├── load_FGNN_data.py
  ├── README.md
  ├── requirements.txt
  ├── run_tu.py
  ├── utils.py
```

## Prepare Dataset

You can choose to obtain dataset from the FGNN2 data processing or directly unzip the compressed package we provided.

## Model training

To train the model for signal probability prediction, use the following command:
```
python run_tu.py  # use default config
```

## Cite Widegate
If Widegate could help your project, please cite our work:
```
@inproceedings{liu2025widegate,
  title={WideGate: Beyond directed acyclic graph learning in subcircuit boundary prediction},
  author={Liu, Jiawei and Liu, Zhiyan and He, Xun and Zhai, Jianwang and Shi, Zhengyuan and Xu, Qiang and Yu, Bei and Shi, Chuan},
  booktitle={2025 Design, Automation \& Test in Europe Conference (DATE)},
  pages={1--7},
  year={2025},
  organization={IEEE}
}
```

