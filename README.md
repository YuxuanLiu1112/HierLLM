# HierLLM: Hierarchical Large Language Model for Question Recommendation


```
Question recommendation is a task that sequentially recommends questions for students to enhance their learning efficiency. Previous methods typically model the question recommendation as a sequential decision-making problem, estimating students’ learning state with the learning history, and feeding the learning state with the learning target to a neural network to select the recommended question from a question set. However, previous methods are faced with two challenges: (1) learning history is unavailable in the cold start scenario, which makes the recommender generate inappropriate recommendations; (2) the size of the question set is much large, which makes it difficult for the recommender to select the best question precisely. To address the challenges, we propose a method called hierarchical large language model for question recommendation (HierLLM), which is a LLM-based hierarchical model. HierLLM tackle the cold start issue with the strong reasoning abilities of LLM, and narrows the range of selectable questions via the hierarchical structure. Comprehensive and in-depth experiments demonstrate the outstanding performance of HierLLM. Our code will be released on https://anonymous.4open.science/r/HierLLM-6481/.
```


## Environment Setup

```
conda env create -f environment.yml
conda activate HierLLM
pip install -e .
```



## Training Simulators 

To pretrain the I-A Simulator（trained using IEKT on Assist09）, run

```
python git_envs/iekt_assist09/main.py
```


## Training and Evaluating HierLLM on IA Simulator

To train and evaluate HierLLM on I-A Simulator, run

```
python main.py --env=default='IEKT_assist09
```
