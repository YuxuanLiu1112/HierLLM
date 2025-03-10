import torch
import random
from tqdm import tqdm
import numpy as np
import os
import inspect
import matplotlib.pyplot as plt
import pickle
from .Reward import GreedyExpReward
from utils import build_knowledge_to_questions_mapping



def select_targets_from_keys(data_dict, selected_keys, target_num):
    combined_list = []
    for key in selected_keys:
        combined_list.extend(data_dict[key])
    
    # 如果 combined_list 中元素数量少于目标数量，则调整目标数量
    target_num = min(target_num, len(combined_list))
    
    return combined_list, target_num

def evenly_select_targets(data_dict, selected_keys, target_num, combined_list):
    selected_indices = []
    key_length = len(selected_keys)
    per_key_selection = target_num // key_length
    
    for key in selected_keys:
        current_list = data_dict[key]
        current_selection = min(per_key_selection, len(current_list))
        selected_samples = random.sample(current_list, current_selection)
        selected_indices.extend(selected_samples)
    
    remaining_selection = target_num - len(selected_indices)
    
    if remaining_selection > 0:
        additional_selection = random.sample([item for item in combined_list if item not in selected_indices], remaining_selection)
        selected_indices.extend(additional_selection)
    
    return selected_indices

class BatchDKTjunyiSimulator:
    def __init__(self, args):

        self.steps = args.steps
        self.episodes = args.episodes
        self.init_records_len = args.init_records_len
        self.device = torch.device(args.env_device)
        self.h = None
        self.epoch_num = args.epoch_num
        self.ques_num = args.ques_num + 1
        self.ques_list = [i for i in range(self.ques_num)]

        for frame_info in inspect.stack():
            if 'batch_DKT_junyi_framework.py' in frame_info.filename:
                self.path = os.path.dirname(frame_info.filename)
        self.env = torch.load(os.path.abspath(self.path+'/mod_dkt_simulator.pt'),
                              map_location=self.device)
        self.env.device = self.device
        self.reward = GreedyExpReward()
        self.target_num = args.target_num

        with open('./data/junyi_ques2know.pkl', 'rb') as file:
            self.junyi_data = pickle.load(file)
            
        # 知识点对应的题目 知识点：[题目]     
        self.junyi_knowledge_to_questions = build_knowledge_to_questions_mapping('junyi', self.junyi_data)


    def batch_train(self, batch_size, agent):
        self.agent = agent
        max_reward = -1

        for epoch_id in range(self.epoch_num):
            self.agent.begin_epoch()
            epoch_reward_list = []
            with tqdm(total=int(self.episodes / self.epoch_num), desc='Iteration %d' % epoch_id) as pbar:
                for i_episode in range(int(self.episodes / self.epoch_num)):

                    batch_target = []  # batch_size个元素，每个元素代表当前env的target
                    # 一、初始化每一个环境
                    self.h, _ = self.env.reset(batch_size)
                    # 生成历史答题记录，交给agent初始化
                    # 1. 初始化学生模型
                    exercises_record = []
                    agent_exercises_record = []

                    # batch_size个学生的历史学习记录
                    for step_id in range(self.init_records_len):
                        batch_init_ques_id = torch.randint(1, self.ques_num, (batch_size,)).to(self.device)
                        self.h, batch_observation, _, _ = self.env.step(self.h, batch_init_ques_id)
                        exercises_record.append(torch.stack([batch_init_ques_id, batch_observation], dim=1))
                        agent_exercises_record.append(torch.stack([batch_init_ques_id - 1, batch_observation], dim=1))
                    # exercises_record: self.init_records_len个元素的列表，每个元素是batch_size长的[ques_id, correct]对列表
                    # 问题下标从1开始
                    batch_init_exercises_record = torch.stack(exercises_record, dim=1)  # batch_size*self.init_records_len*2
                    # 问题下标从0开始
                    agent_batch_init_exercises_record = torch.stack(agent_exercises_record, dim=1).tolist()  # batch_size*self.init_records_len*2
                    # # 2. 初始化学习目标
                    # # ------------根据知识点选择目标----------
                    # data_dict = self.junyi_knowledge_to_questions
                    # # 选出10个随机key
                    # selected_keys = random.sample(list(data_dict.keys()), 10)
                    
                    # combined_list = []
                    # for key in selected_keys:
                    #     combined_list.extend(data_dict[key])
                        
                    # if len(combined_list) < self.target_num:
                    #     self.target_num = len(combined_list)

                    # batch_target_table = torch.full((batch_size, len(combined_list)), False).to(self.device)
                    # target = torch.full((batch_size, self.target_num), -1, dtype=torch.long).to(self.device)

                    # for i in range(batch_size):
                    #     key_length = len(selected_keys)
                        
                    #     # 每个key均匀选择
                    #     per_key_selection = self.target_num // key_length
                        
                    #     selected_indices_for_batch = []
                        
                    #     for key in selected_keys:
                    #         current_list = data_dict[key]
                    #         current_selection = min(per_key_selection, len(current_list))
                    #         selected_samples = random.sample(current_list, current_selection)
                    #         selected_indices_for_batch.extend(selected_samples)
                        
                    #     remaining_selection = self.target_num - len(selected_indices_for_batch)
                        
                    #     # 补充剩余的选择
                    #     if remaining_selection > 0:
                    #         additional_selection = random.sample([item for item in combined_list if item not in selected_indices_for_batch], remaining_selection)
                    #         selected_indices_for_batch.extend(additional_selection)
                        
                    #     # 将选出的 indices 填入 target
                    #     target[i] = torch.tensor(selected_indices_for_batch).to(self.device)

                    # # 将选择的目标设置为 True
                    # for i in range(batch_size):
                    #     for index in target[i]:
                    #         batch_target_table[i, combined_list.index(index.item())] = True
                    
                    # batch_target = (target-1).tolist()

                    
                    #-----------------------------------
                    target_prob = torch.randn(batch_size, self.ques_num-1).to(self.device)

                    # 以下两个均为batch_size * self.target_num矩
                    target = torch.topk(target_prob, k=self.target_num)[1] + 1
                    batch_target = (target-1).tolist()
                    index = torch.arange(0, batch_size).repeat(self.target_num, 1).T

                    batch_target_table = torch.full((batch_size, self.ques_num), False).to(self.device)
                    batch_target_table[index, target] = torch.full((batch_size, self.target_num), True).to(self.device)
                    # ----------------------------------


                    # 3. 获取初始分数
                    batch_score_init = self.env.test_target_score(target)

                    self.agent.begin_episode(agent_batch_init_exercises_record, batch_target)

                    # 二、每个环境都做steps步交互
                    for step in range(self.steps):
                        ques_id_list = self.agent.take_action()
                        batch_ques_id = torch.tensor(ques_id_list, device=self.device) + 1
                        self.h, batch_observation, _, _ = self.env.step(self.h, batch_ques_id)

                        observation_list = batch_observation.tolist()

                        self.agent.step_refresh(ques_id_list, observation_list)

                    # 三、归一化的进步值作为奖励
                    batch_score_aft = self.env.test_target_score(target)
                    # all_score = self.env.total_probability
                    # # batch_size * ques_num, 是学习目标的位置标记学生得分，否则置为0
                    # batch_target_score_table = torch.where(batch_target_table, all_score, 0)
                    # # batch_size长的向量，标记每个学生的初始得分
                    # batch_score_aft = torch.sum(batch_target_score_table, dim=1)
                    # batch_size长的向量，记录每个学生的学习效果
                    batch_reward = (batch_score_aft - batch_score_init) / (self.target_num - batch_score_init)
                    epoch_reward_list.append(torch.mean(batch_reward).item())

                    self.agent.episode_refresh(batch_reward, init_score=batch_score_init, aft_score=batch_score_aft,
                                               full_score=self.target_num, terminal_tag=False)

                    # 更新进度条
                    pbar.set_postfix({
                        'episode':
                            '%d' % (self.episodes / 10 * epoch_id + i_episode + 1),
                        'ave_score_after':
                            '%.6f' % torch.mean(batch_reward)
                    })
                    pbar.update(1)

                    this_reward = torch.mean(batch_reward)
                    if this_reward > max_reward:
                        max_reward = this_reward
                        if self.agent.name == 'CSEAL':
                            from mxnet import ndarray
                            net = self.agent.agent.value_net.net_mod.net
                            params = net._collect_params_with_prefix()
                            arg_dict = {key: val._reduce() for key, val in params.items()}
                            ndarray.save('best_agent/DKTjunyi/{}.parmas'.format(self.agent.name), arg_dict)
                        elif self.agent.name == 'SRC':
                            from mindspore import save_checkpoint
                            save_checkpoint(self.agent.model, 'best_agent/DKTjunyi/{}.ckpt'.format(self.agent.name))
                        else:
                            # torch.save(self.agent, 'best_agent/DKTjunyi/{}.pt'.format(self.agent.name))
                            output_dir = 'best_agent/DKTjunyi/{}'.format(self.agent.name)
            
                            # self.agent.model.llama_model.save_pretrained(output_dir)
                            self.agent.model.llama_model_H.save_pretrained(output_dir)
                            self.agent.model.llama_model_L.save_pretrained(output_dir)
                            model_path = os.path.join(output_dir, "adapter.pth")
                            
                            know_embedding = self.agent.model.know_embedding.state_dict()
                            ques_embedding = self.agent.model.ques_embedding.state_dict()
                            emb_to_hidden = self.agent.model.emb_to_hidden.state_dict()
                            emb_to_double_hidden = self.agent.model.emb_to_double_hidden.state_dict()
                            encoder =self.agent.model.encoder.state_dict()
                            kt_mlp = self.agent.model.kt_mlp.state_dict()
                            ques_correct_to_hidden = self.agent.model.kt_mlp.state_dict()
                            init_state_encoder = self.agent.model.init_state_encoder.state_dict()
                            state_encoder = self.agent.model.state_encoder.state_dict()
                            seq_encoder = self.agent.model.seq_encoder.state_dict()
                            norm = self.agent.model.norm.state_dict()
                            sigmoid = self.agent.model.sigmoid.state_dict()
                            
                            W0 = self.agent.model.W0.state_dict()
                            W1 = self.agent.model.W1.state_dict()
                            W2 = self.agent.model.W2.state_dict()
                            W3 = self.agent.model.W3.state_dict()
                            vt = self.agent.model.vt.state_dict()
                            know_input_proj,input_proj,know_score,score = self.agent.model.know_input_proj.state_dict(), self.agent.model.input_proj.state_dict(), self.agent.model.know_score.state_dict(), self.agent.model.score.state_dict()
                            
                            torch.save(
                                {
                                    'know_embedding': know_embedding,
                                    'ques_embedding': ques_embedding,
                                    'emb_to_hidden': emb_to_hidden,
                                    'emb_to_double_hidden': emb_to_double_hidden,
                                    'encoder': encoder,
                                    'kt_mlp': kt_mlp,
                                    'ques_correct_to_hidden': ques_correct_to_hidden,
                                    'init_state_encoder': init_state_encoder,
                                    'vt': vt,
                                    'norm': norm,
                                    'sigmoid': sigmoid,
                                    'state_encoder': state_encoder,
                                    'W0': W0,
                                    'W1': W1,
                                    'W2': W2,
                                    'W3': W3,
                                    'know_input_proj': know_input_proj,
                                    'input_proj': input_proj,
                                    'know_score': know_score,
                                    'score': score,
                                }, model_path)
            self.agent.epoch_refresh()
            print(epoch_reward_list)
        if self.agent.name == 'CSEAL':
            best_agent = None
        elif self.agent.name == 'SRC':
            best_agent = None
        else:
            # best_agent = torch.load('best_agent/DKTjunyi/{}.pt'.format(self.agent.name), map_location=self.agent.device)
            pass
        # return self.agent, best_agent, max_reward
        return self.agent, max_reward


    def batch_test(self, batch_size, agent, test_times=100):
        self.agent = agent

        self.agent.begin_epoch()

        batch_ave_reward_list = []

        with tqdm(total=test_times) as pbar:
            for i_episode in range(test_times):
                batch_target = []  # batch_size个元素，每个元素代表当前env的target
                # 一、初始化每一个环境
                self.h, _ = self.env.reset(batch_size)
                # 生成历史答题记录，交给agent初始化
                # 1. 初始化学生模型
                exercises_record = []
                agent_exercises_record = []

                # batch_size个学生的历史学习记录
                for step_id in range(self.init_records_len):
                    batch_init_ques_id = torch.randint(1, self.ques_num, (batch_size,)).to(self.device)
                    self.h, batch_observation, _, _ = self.env.step(self.h, batch_init_ques_id)
                    exercises_record.append(torch.stack([batch_init_ques_id, batch_observation], dim=1))
                    agent_exercises_record.append(torch.stack([batch_init_ques_id - 1, batch_observation], dim=1))
                # exercises_record: self.init_records_len个元素的列表，每个元素是batch_size长的[ques_id, correct]对列表
                batch_init_exercises_record = torch.stack(exercises_record, dim=1)  # batch_size*self.init_records_len*2
                agent_batch_init_exercises_record = torch.stack(agent_exercises_record, dim=1)

                # 2. 初始化学习目标
                target_prob = torch.randn(batch_size, self.ques_num-1).to(self.device)

                # 以下两个均为batch_size * self.target_num矩
                target = torch.topk(target_prob, k=self.target_num)[1] + 1
                batch_target = (target - 1).tolist()
                index = torch.arange(0, batch_size).repeat(self.target_num, 1).T

                batch_target_table = torch.full((batch_size, self.ques_num), False).to(self.device)
                batch_target_table[index, target] = torch.full((batch_size, self.target_num), True).to(self.device)

                # 3. 获取初始分数
                batch_score_init = self.env.test_target_score(target)

                self.agent.begin_episode(agent_batch_init_exercises_record.tolist(), batch_target)

                # 二、每个环境都做steps步交互
                for step in range(self.steps):
                    ques_id_list = self.agent.take_action()
                    batch_ques_id = torch.tensor(ques_id_list, device=self.device) + 1
                    self.h, batch_observation, _, _ = self.env.step(self.h, batch_ques_id)
                    observation_list = batch_observation.tolist()
                    self.agent.test_step_refresh(ques_id_list, observation_list)

                # 三、归一化的进步值作为奖励
                batch_score_aft = self.env.test_target_score(target)

                # all_score = self.env.total_probability
                # # batch_size * ques_num, 是学习目标的位置标记学生得分，否则置为0
                # batch_target_score_table = torch.where(batch_target_table, all_score, 0)
                # # batch_size长的向量，标记每个学生的初始得分
                # batch_score_aft = torch.sum(batch_target_score_table, dim=1)
                # batch_size长的向量，记录每个学生的学习效果
                batch_reward = (batch_score_aft - batch_score_init) / (self.target_num - batch_score_init)

                batch_ave_reward = torch.mean(batch_reward)
                batch_ave_reward_list.append(batch_ave_reward.item())

                self.agent.test_episode_refresh()

                # 更新进度条
                pbar.set_postfix({
                    'episode':
                        '%d' % (i_episode + 1),
                    'ave_score_after':
                        '%.6f' % torch.mean(batch_reward).item()
                })
                pbar.update(1)

        test_mean_reward = sum(batch_ave_reward_list) / len(batch_ave_reward_list)

        return test_mean_reward
