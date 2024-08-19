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


class BatchIEKTassist09Simulator:
    def __init__(self, args):
        self.agent = None

        self.steps = args.steps
        self.episodes = args.episodes
        self.init_records_len = args.init_records_len
        self.device = torch.device(args.env_device)
        self.h = None
        self.epoch_num = args.epoch_num
        self.ques_num = args.ques_num + 1
        self.ques_list = [i for i in range(self.ques_num)]

        for frame_info in inspect.stack():
            if 'batch_iekt_assist09_framework.py' in frame_info.filename:
                self.path = os.path.dirname(frame_info.filename)
        self.env = torch.load(os.path.abspath(self.path+'/iekt_assist09_simulator.pt'),
                              map_location=self.device)
        self.env.device = self.device
        self.reward = GreedyExpReward()
        self.target_num = args.target_num

        with open('./data/assist_ques2know.pkl', 'rb') as file:
            self.assist_data = pickle.load(file)
                
        self.assist_knowledge_to_questions = build_knowledge_to_questions_mapping('assist', self.assist_data)


    def batch_train(self, batch_size, agent):
        self.agent = agent
        max_reward = -1

        for epoch_id in range(self.epoch_num):
            self.agent.begin_epoch()
            epoch_reward_list = []
            with tqdm(total=int(self.episodes / self.epoch_num), desc='Iteration %d' % epoch_id) as pbar:
                for i_episode in range(int(self.episodes / self.epoch_num)):

                    batch_target = []  

                    self.h, _ = self.env.reset(batch_size)
                    exercises_record = []
                    agent_exercises_record = []

                    for step_id in range(self.init_records_len):
                        batch_init_ques_id = torch.randint(1, self.ques_num, (batch_size,)).to(self.device)
                        self.h, batch_observation, _, _ = self.env.step(self.h, batch_init_ques_id)
                        exercises_record.append(torch.stack([batch_init_ques_id, batch_observation], dim=1))
                        agent_exercises_record.append(torch.stack([batch_init_ques_id - 1, batch_observation], dim=1))
                  
                    batch_init_exercises_record = torch.stack(exercises_record, dim=1)  # batch_size*self.init_records_len*2
                  
                    agent_batch_init_exercises_record = torch.stack(agent_exercises_record, dim=1).tolist()  # batch_size*self.init_records_len*2
                    
                    target_prob = torch.randn(batch_size, self.ques_num-1).to(self.device)

                    target = torch.topk(target_prob, k=self.target_num)[1] + 1
                    batch_target = (target-1).tolist()
                    index = torch.arange(0, batch_size).repeat(self.target_num, 1).T

                    batch_target_table = torch.full((batch_size, self.ques_num), False).to(self.device)
                    batch_target_table[index, target] = torch.full((batch_size, self.target_num), True).to(self.device)
                    
                    batch_score_init = self.env.test_target_score(target)

                    self.agent.begin_episode(agent_batch_init_exercises_record, batch_target)

      
                    for step in range(self.steps):
                        ques_id_list = self.agent.take_action()
                        batch_ques_id = torch.tensor(ques_id_list, device=self.device) + 1
                        self.h, batch_observation, _, _ = self.env.step(self.h, batch_ques_id)

                        observation_list = batch_observation.tolist()

                        self.agent.step_refresh(ques_id_list, observation_list)

        
                    batch_score_aft = self.env.test_target_score(target)
                    
                    batch_reward = (batch_score_aft - batch_score_init) / (self.target_num - batch_score_init)
                    epoch_reward_list.append(torch.mean(batch_reward).item())

                    self.agent.episode_refresh(batch_reward, init_score=batch_score_init, aft_score=batch_score_aft,
                                               full_score=self.target_num, terminal_tag=False)

               
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
                            ndarray.save('best_agent/IEKTassist09/{}.parmas'.format(self.agent.name), arg_dict)
                       
                        else:
                            #
                            output_dir = 'best_agent/IEKTassist09/{}'.format(self.agent.name)
            
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
        else:
            pass
        return self.agent, max_reward


    def batch_test(self, batch_size, agent, test_times=100):
        self.agent = agent

        self.agent.begin_epoch()

        batch_ave_reward_list = []

        with tqdm(total=test_times) as pbar:
            for i_episode in range(test_times):
                batch_target = []  
                self.h, _ = self.env.reset(batch_size)
                exercises_record = []
                agent_exercises_record = []

                for step_id in range(self.init_records_len):
                    batch_init_ques_id = torch.randint(1, self.ques_num, (batch_size,)).to(self.device)
                    self.h, batch_observation, _, _ = self.env.step(self.h, batch_init_ques_id)
                    exercises_record.append(torch.stack([batch_init_ques_id, batch_observation], dim=1))
                    agent_exercises_record.append(torch.stack([batch_init_ques_id - 1, batch_observation], dim=1))
                batch_init_exercises_record = torch.stack(exercises_record, dim=1)  # batch_size*self.init_records_len*2
                agent_batch_init_exercises_record = torch.stack(agent_exercises_record, dim=1)

                
                target_prob = torch.randn(batch_size, self.ques_num-1).to(self.device)

                
                target = torch.topk(target_prob, k=self.target_num)[1] + 1
                batch_target = (target - 1).tolist()
                index = torch.arange(0, batch_size).repeat(self.target_num, 1).T

                batch_target_table = torch.full((batch_size, self.ques_num), False).to(self.device)
                batch_target_table[index, target] = torch.full((batch_size, self.target_num), True).to(self.device)

                batch_score_init = self.env.test_target_score(target)

                self.agent.begin_episode(agent_batch_init_exercises_record.tolist(), batch_target)

                for step in range(self.steps):
                    ques_id_list = self.agent.take_action()
                    batch_ques_id = torch.tensor(ques_id_list, device=self.device) + 1
                    self.h, batch_observation, _, _ = self.env.step(self.h, batch_ques_id)
                    observation_list = batch_observation.tolist()
                    self.agent.test_step_refresh(ques_id_list, observation_list)

                batch_score_aft = self.env.test_target_score(target)

                batch_reward = (batch_score_aft - batch_score_init) / (self.target_num - batch_score_init)

                batch_ave_reward = torch.mean(batch_reward)
                batch_ave_reward_list.append(batch_ave_reward.item())

                self.agent.test_episode_refresh()

                pbar.set_postfix({
                    'episode':
                        '%d' % (i_episode + 1),
                    'ave_score_after':
                        '%.6f' % torch.mean(batch_reward).item()
                })
                pbar.update(1)

        test_mean_reward = sum(batch_ave_reward_list) / len(batch_ave_reward_list)

        return test_mean_reward

    def train(self, agent):
        self.agent = agent
        return_list = []

        for i in range(self.epoch_num):
            with tqdm(total=int(self.episodes / self.epoch_num), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.episodes / self.epoch_num)):
                    self.h, total_probability = self.env.reset()

                    exercises_record = []
                    for _ in range(self.init_records_len):
                        ques_id = random.randint(0, 2163)
                        self.h, observation, reward, score_aft = self.env.step(self.h, ques_id)
                        exercises_record.append([ques_id, observation])

                    target_num = random.randint(300, 500)
                    target = set(random.sample(self.ques_list, target_num))

                    all_score = self.env.total_probability[0]
                    target_score_init = 0
                    for target_id in target:
                        target_score_init += all_score[target_id].item()

                    self.agent.initialize(exercises_record, target)

                    for step in range(self.steps):
                        try:
                            ques_id = self.agent.take_action()
                        except StopIteration:
                            break

                        self.h, observation, reward, score_aft = self.env.step(self.h, ques_id)
                        self.agent.step_refresh(ques_id, observation)

                    all_score = self.env.total_probability[0]
                    target_score_aft = 0
                    for target_id in target:
                        target_score_aft += all_score[target_id].item()

                    return_value = (target_score_aft - target_score_init)/(len(target) - target_score_init)
                    return_list.append(return_value)  
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({
                            'episode':
                                '%d' % (self.episodes / 10 * i + i_episode + 1),
                            'return':
                                '%.3f' % np.mean(return_list[-10:])
                        })

                    reward_values = self.reward(
                        initial_score=target_score_init,
                        final_score=target_score_aft,
                        full_score=len(target),
                        path=self.agent.path,
                        terminal_tag=False,
                    )

                    self.agent.episode_refresh(return_value, reward_values)

                    pbar.update(1)

                self.agent.epoch_refresh()

        episode_list = [i for i in range(self.episodes)]
        plt.plot(episode_list, return_list)
        plt.xlabel('episode')
        plt.ylabel('final_score')

        

        return self.agent, return_list

    def test(self, agent, test_times=100):
        self.agent = agent
        score_list = []
        growth_list = []
        return_list = []

        with tqdm(total=test_times) as pbar:
            for i_episode in range(int(test_times)):
                self.h, total_probability = self.env.reset()
                exercises_record = []
                for _ in range(self.init_records_len):
                    ques_id = random.randint(0, 2163)
                    self.h, observation, reward, score_aft = self.env.step(self.h, ques_id)
                    exercises_record.append([ques_id, observation])

                target_num = random.randint(300, 500)
                target = set(random.sample(self.ques_list, target_num))

                all_score = self.env.total_probability[0]
                target_score_init = 0
                for target_id in target:
                    target_score_init += all_score[target_id].item()

                self.agent.initialize(exercises_record, target)

                for step in range(self.steps):
                    ques_id = self.agent.take_action()
                    self.h, observation, reward, score_aft = self.env.step(self.h, ques_id)

                    self.agent.step_refresh_test(ques_id, observation)

                all_score = self.env.total_probability[0]
                target_score_aft = 0
                for target_id in target:
                    target_score_aft += all_score[target_id].item()

                return_value = (target_score_aft - target_score_init) / (len(target) - target_score_init)
                return_list.append(return_value)  
                pbar.set_postfix({
                    'episode':
                        '%d' % (i_episode + 1),
                    'return':
                        '%.3f' % return_list[-1]
                })

                self.agent.episode_refresh_test()

                pbar.update(1)

        mean_reward = np.mean(return_list)

        return mean_reward

    def train_for_rltutor(self, agent):
        self.agent = agent
        score_list = []

        self.h, total_probability = self.env.reset()

        exercises_record = []
        for _ in range(self.init_records_len):
            ques_id = random.randint(0, self.ques_num)
            self.h, observation, reward, score_aft = self.env.step(self.h, ques_id)
            exercises_record.append([ques_id, observation])
        self.agent.initialize(exercises_record)

        for step in range(20):
            for _ in range(10):
                ques_id = self.agent.take_action()
                self.h, observation, reward, score_aft = self.env.step(self.h, ques_id)
                score_list.append(score_aft)
                self.agent.step(ques_id, observation.item())

            self.agent.save_and_refresh()

        return score_list

    def test_for_generator(self, agent, max_steps):
        self.agent = agent
        score_list = []
        score_aft = 0

        self.h, total_probability = self.env.reset()

        exercises_record = []
        for _ in range(self.init_records_len):
            ques_id = random.randint(0, self.ques_num)
            self.h, observation, reward, score_aft = self.env.step(self.h, ques_id)
            exercises_record.append([ques_id, observation])
        self.agent.initialize(exercises_record)

        for step in range(max_steps):
            ques_id = self.agent.take_action()
            self.h, observation, reward, score_aft = self.env.step(self.h, ques_id)
            score_list.append(score_aft)
            self.agent.step(ques_id, observation.item())

        reward = score_aft
        sa_record = self.agent.sa_record

        return reward, sa_record
