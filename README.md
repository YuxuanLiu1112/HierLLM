# HQRA: Hierarchical Question Recommendation Agent


## Environment Setup

```
conda env create -f environment.yml
conda activate HQRA
pip install -e .
```



## Training Simulators 

To pretrain the D-J Simulator（trained using DKT on junyi）, run

```
python git_envs/dkt_junyi/main.py
```


## Training and Evaluating HQRA on D-J Simulator

To train and evaluate HQRA on D-J Simulator, run

```
python main.py --env=default='DKT_junyi'
```
