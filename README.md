# HierLLM: Hierarchical Large Language Model for Question Recommendation


## Environment Setup

```
conda env create -f environment.yml
conda activate HierLLM
pip install -e .
```



## Training Simulators 

To pretrain the D-J Simulator（trained using DKT on junyi）, run

```
python git_envs/dkt_junyi/main.py
```


## Training and Evaluating HierLLM on D-J Simulator

To train and evaluate HierLLM on D-J Simulator, run

```
python main.py --env=default='DKT_junyi'
```
