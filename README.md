# HQRA: Hierarchical Question Recommendation Agent


## Environment Setup

```
conda env create -f environment.yml
conda activate HQRA
pip install -e .
```



## Training Simulators 

To pretrain the I-A Simulator（trained using IEKT on Assist09）, run

```
python git_envs/iekt_assist09/main.py
```


## Training and Evaluating HQRA on I-A Simulator

To train and evaluate HQRA on I-A Simulator, run

```
python main.py --env=default='IEKT_assist09
```
