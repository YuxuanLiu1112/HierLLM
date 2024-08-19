import random

import torch
import torch.nn as nn
from torch.distributions import Categorical

from utils import Encoder, self_attention, build_knowledge_to_questions_mapping, find_related_questions_for_list
from collections import OrderedDict
import copy
import pickle

import transformers
from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

# from prompter import Prompter

from peft import(
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)


class HierLLMModel(nn.Module):
    def __init__(self, batch_size, ques_num, emb_dim, hidden_dim, weigh_dim, target_num,
                 policy_mlp_hidden_dim_list, kt_mlp_hidden_dim_list, use_kt, n_steps, n_head, n_layers, n_ques,
                 device, label, know_num, m=200, rank_num=10):
        super(HierLLMModel, self).__init__()

        self.batch_size = batch_size
        self.ques_num = ques_num
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.weigh_dim = weigh_dim
        self.use_kt = use_kt
        self.n_steps = n_steps
        self.n_ques = n_ques

        self.know_embedding = nn.Embedding(know_num, emb_dim, device=device)
        self.ques_embedding = nn.Embedding(ques_num, emb_dim, device=device)
        self.emb_to_hidden = nn.Linear(emb_dim, hidden_dim, device=device)
        self.emb_to_double_hidden = nn.Linear(emb_dim, hidden_dim*2, device=device)

        self.encoder = Encoder(hidden_dim, hidden_dim, n_head=n_head, n_layers=n_layers, drop_prob=0.5).to(device)

        self.kt_mlp = nn.Sequential(
            OrderedDict([
                (
                    'layer{}'.format(i),
                    nn.Linear(kt_mlp_hidden_dim_list[i], kt_mlp_hidden_dim_list[i + 1], device=device)
                 )
                for i in range(len(kt_mlp_hidden_dim_list) - 1)
            ])
        )

        self.ques_correct_to_hidden = nn.Linear(emb_dim+1, hidden_dim).to(device)
        self.init_state_encoder = nn.LSTM(hidden_dim, hidden_dim * 2, batch_first=True).to(device)
        self.state_encoder = nn.LSTM(hidden_dim * 2, hidden_dim * 2, batch_first=True).to(device)

        self.seq_encoder = nn.LSTM(ques_num, hidden_dim).to(device)

        self.norm = nn.BatchNorm1d(ques_num).to(device)
        self.sigmoid = nn.Sigmoid()

        self.raw_ques_embedding = None
        self.ques_representation = None
        self.batch_target_emb = None
        self.hc = None
        self.batch_state = None
        self.state_encoder_inputs = None

        self.last_ques = None
        self.last_n_ques = None

        self.action_prob_list = []
        self.kt_prob_list = []
        self.action_list = []
        
        self.device = device

        self.target = None
        self.target_num = target_num
        self.batch_target_num = None
        self.batch_T_rep = None
        self.target_table = None
        self.m = m
        self.ranking_loss = 0
        self.rank_num = rank_num

        with open('./data/assist_ques2know.pkl', 'rb') as file:
            self.assist_data = pickle.load(file)
        with open('./data/junyi_ques2know.pkl', 'rb') as file:
            self.junyi_data = pickle.load(file)
        
        self.assist_knowledge_to_questions = build_knowledge_to_questions_mapping('assist', self.assist_data)
        self.junyi_knowledge_to_questions = build_knowledge_to_questions_mapping('junyi', self.junyi_data)

        if label == "assist":
                self.ques2know = self.assist_data
                self.knowledge_to_questions = self.assist_knowledge_to_questions
        elif label == 'junyi':
                self.ques2know = self.junyi_data
                self.knowledge_to_questions = self.junyi_knowledge_to_questions
        
        # LLM 
        self.llm_output_dim = ques_num
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0
        self.lora_target_modules = ["q_proj", "v_proj",]

        self.base_model = '../llama-7b-hf'
        self.cache_dir = "./weights/llama"
        self.device_map = 'auto'

        # self.instruction_text, self.instruction_text_ = Prompter.generate_prompt(self, task_type='sequential')
        
        print('Initializing language decoder ...')

        # add the lora module
        peft_config_H = LoraConfig(
                task_type='FEATURE_EXTRACTION',
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                bias='none',
        )
        peft_config_L = LoraConfig(
                task_type='FEATURE_EXTRACTION',
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                bias='none',
        )
        self.llama_model_H = LlamaModel.from_pretrained(self.base_model, load_in_8bit=True, torch_dtype=torch.float16,
                                        local_files_only=True, cache_dir=self.cache_dir, device_map=self.device_map)
        self.llama_model_H = prepare_model_for_kbit_training(self.llama_model_H)
        self.llama_model_H = get_peft_model(self.llama_model_H, peft_config_H)
        self.llama_model_H.print_trainable_parameters()
        self.llama_model_H.config.use_cache = False
        
        self.llama_tokenizer_H = LlamaTokenizer.from_pretrained(self.base_model, use_fast=False, local_files_only=True, cache_dir=self.cache_dir)
        self.llama_tokenizer_H.pad_token_id = 0
        self.llama_tokenizer_H.padding_side = 'right'
        
        self.llama_model_L = LlamaModel.from_pretrained(self.base_model, load_in_8bit=True, torch_dtype=torch.float16,
                                        local_files_only=True, cache_dir=self.cache_dir, device_map=self.device_map)
        self.llama_model_L = prepare_model_for_kbit_training(self.llama_model_L)
        self.llama_model_L = get_peft_model(self.llama_model_L, peft_config_L)
        self.llama_model_L.print_trainable_parameters()
        self.llama_model_L.config.use_cache = False

        self.llama_tokenizer_L = LlamaTokenizer.from_pretrained(self.base_model, use_fast=False, local_files_only=True, cache_dir=self.cache_dir)
        self.llama_tokenizer_L.pad_token_id = 0
        self.llama_tokenizer_L.padding_side = 'right'

        # Prompt
        self.L_instruction_prompt = "Predict the next question."
        self.H_instruction_prompt = "Predict the next concept."
        self.student_prompt = "### Instruction: Based on students' learning history,"
        self.last_learned_ques_prompt = "and questions"
        self.target_prompt = "learning targets"
        self.know_prompt = "and concepts"
        self.ques_prompt = "and questions"
        self.response = "### Response:"
        
        self.ques_prompt_ids, self.ques_prompt_mask = self.llama_tokenizer_L(self.ques_prompt, 
                                                        truncation=False, padding=False,
                                                        return_tensors='pt', add_special_tokens=False).values()
        
        self.know_prompt_ids, self.know_prompt_mask = self.llama_tokenizer_L(self.know_prompt, 
                                                        truncation=False, padding=False,
                                                        return_tensors='pt', add_special_tokens=False).values()
        
        self.student_prompt_ids, self.student_prompt_mask = self.llama_tokenizer_L(self.student_prompt, 
                                                        truncation=False, padding=False,
                                                        return_tensors='pt', add_special_tokens=False).values()
        self.last_learned_ques_ids, self.last_learned_ques_mask = self.llama_tokenizer_L(self.last_learned_ques_prompt, 
                                                        truncation=False, padding=False,
                                                        return_tensors='pt', add_special_tokens=False).values()
        self.target_prompt_ids, self.target_prompt_mask = self.llama_tokenizer_L(self.target_prompt, 
                                                        truncation=False, padding=False,
                                                        return_tensors='pt', add_special_tokens=False).values()
        

        self.L_instruct_ids, self.L_instruct_mask = self.llama_tokenizer_L(self.L_instruction_prompt, 
                                                                    truncation=True, padding=False,
                                                                    return_tensors='pt', add_special_tokens=False).values()
        self.H_instruct_ids, self.H_instruct_mask = self.llama_tokenizer_L(self.H_instruction_prompt, 
                                                                    truncation=True, padding=False,
                                                                    return_tensors='pt', add_special_tokens=False).values()
        self.response_ids, self.response_mask = self.llama_tokenizer_L(self.response,
                                                                        truncation=True, padding=False,
                                                                        return_tensors='pt', add_special_tokens=False).values()

        self.instruct_embeds_L = self.llama_model_H.model.embed_tokens(self.L_instruct_ids.cuda(0)).expand(self.batch_size, -1, -1).mean(dim=1)
        self.instruct_embeds_H = self.llama_model_H.model.embed_tokens(self.H_instruct_ids.cuda(0)).expand(self.batch_size, -1, -1).mean(dim=1)
        self.response_embeds = self.llama_model_L.model.embed_tokens(self.response_ids.cuda(0)).expand(self.batch_size, -1, -1).mean(dim=1)
        self.know_prompt_embeds = self.llama_model_L.model.embed_tokens(self.know_prompt_ids.cuda(0)).expand(self.batch_size, -1, -1).mean(dim=1)
        self.ques_prompt_embeds = self.llama_model_L.model.embed_tokens(self.ques_prompt_ids.cuda(0)).expand(self.batch_size, -1, -1).mean(dim=1)
    
        self.L_instruct_mask = self.L_instruct_mask.cuda(0).expand(self.batch_size, -1)
        self.H_instruct_mask = self.H_instruct_mask.cuda(0).expand(self.batch_size, -1)
        
        self.response_mask = self.response_mask.cuda(0).expand(self.batch_size, -1)
        
        self.student_embeds = self.llama_model_L.model.embed_tokens(self.student_prompt_ids.cuda(0)).expand(self.batch_size, -1, -1).mean(dim=1)
        self.last_ques_embeds = self.llama_model_L.model.embed_tokens(self.last_learned_ques_ids.cuda(0)).expand(self.batch_size, -1, -1).mean(dim=1)
        self.targets_embeds = self.llama_model_L.model.embed_tokens(self.target_prompt_ids.cuda(0)).expand(self.batch_size, -1, -1).mean(dim=1)

        self.student_mask = self.student_prompt_mask.cuda(0).expand(self.batch_size, -1)
        self.last_ques_mask = self.last_learned_ques_mask.cuda(0).expand(self.batch_size, -1)
        self.targets_mask = self.target_prompt_mask.cuda(0).expand(self.batch_size, -1)
        self.know_prompt_mask = self.know_prompt_mask.cuda(0).expand(self.batch_size, -1)
            
        print('Language decoder initialized.')

        # projection layer
        self.W0 = nn.Linear(hidden_dim*2, self.llama_model_L.config.hidden_size, bias=False).to(device)
        self.W1 = nn.Linear(hidden_dim*2, self.llama_model_L.config.hidden_size, bias=False).to(device)
        self.W2 = nn.Linear(hidden_dim*2, self.llama_model_L.config.hidden_size, bias=False).to(device)
        self.W3 = nn.Linear(hidden_dim * 2, self.llama_model_L.config.hidden_size, bias=False).to(device)
        
        # Linear
        self.vt = nn.Linear(weigh_dim*2, 1, bias=False).to(device)
        self.know_input_proj = nn.Linear(self.llama_model_L.config.hidden_size*8,self.llama_model_L.config.hidden_size).to(self.device)
        self.input_proj = nn.Linear(self.llama_model_L.config.hidden_size*8,self.llama_model_L.config.hidden_size).to(self.device)
        self.know_score = nn.Linear(self.llama_model_L.config.hidden_size, know_num, bias=False).to(self.device)
        self.score = nn.Linear(self.llama_model_L.config.hidden_size, ques_num, bias=False).to(self.device)
                
        # others
        self.know_num = know_num
        self.last_tartet_list = []
        self.last_know_list = []
        self.last_ques_list = []





    def forward(self, ques_id, observation):
        know, know_prob, action, ques_prob = self.take_action()
        kt_prob = self.step_refresh(ques_id, observation)

        return know, know_prob, action, ques_prob, kt_prob

    def initialize(self, exercises_record, targets, batch_size=None):
        targets = copy.deepcopy(targets)

        if not batch_size:
            batch_size = self.batch_size

        self.raw_ques_embedding = None
        self.ques_representation = None
        self.batch_target_emb = None
        self.hc = None
        self.batch_state = None
        self.state_encoder_inputs = None

        self.last_ques = None
        self.last_n_ques = None

        self.kt_prob_list = []
        self.action_prob_list = []
        self.knowledge_prob_list = []
        self.action_list = []
        last_ques_list = []
        last_n_ques_list = []

        self.target_table = torch.zeros(self.batch_size, self.ques_num).to(self.device)


        self.raw_ques_embedding = self.ques_embedding(torch.arange(self.ques_num).to(self.device))
        ques_embedding = self.emb_to_hidden(self.raw_ques_embedding)

        ques_att_embedding = self.encoder(ques_embedding.unsqueeze(0)).squeeze(0)

        self.ques_representation = torch.cat([ques_att_embedding, ques_embedding], dim=1)
        
        self.raw_know_embedding = self.know_embedding(torch.arange(self.know_num).to(self.device))
        know_embedding = self.emb_to_hidden(self.raw_know_embedding)
        know_att_embedding = self.encoder(know_embedding.unsqueeze(0)).squeeze(0)
        self.know_representation = torch.cat([know_att_embedding, know_embedding], dim=1)  
        
        targets_embedding_list = []
        self.batch_target_num = torch.zeros(self.batch_size).to(self.device)


        for i in range(len(targets)):
            target = targets[i]
            all_knowledge_ids = []

            for ques_id in target:
                self.target_table[i, ques_id] = 1
            
            targets_embedding = self.ques_representation[torch.tensor(list(target)).to(self.device)]
            mean_targets_embedding = torch.mean(targets_embedding, dim=0)
            targets_embedding_list.append(mean_targets_embedding)

            self.batch_target_num[i] = len(targets[i])
            targets[i] = list(targets[i]) + [self.target_num for _ in range(self.target_num - len(targets[i]))]

        self.target = torch.tensor(targets, requires_grad=False).to(self.device)  
        self.batch_target_emb = torch.stack(targets_embedding_list).to(self.device) 

        ques_representation = torch.cat(
            [self.ques_representation, torch.zeros(1, self.hidden_dim * 2).to(self.device)], dim=0)
        self.batch_T_rep = ques_representation[self.target]  
        batch_init_h = []
        batch_init_c = []

        for exercise_record in exercises_record:
            exercise_record = torch.tensor(exercise_record).to(self.device)
            ques_ids = exercise_record[:, 0]

            last_ques_id = ques_ids[-1]
            try:
                last_n_ques_id = ques_ids[-self.n_ques:]
            except AttributeError:
                last_n_ques_id = ques_ids[-1:]

            last_ques_list.append(last_ques_id.item())
            last_n_ques_list.append(last_n_ques_id)

            raw_ques_embedding = self.raw_ques_embedding[ques_ids]
            corrects = exercise_record[:, 1].view(-1, 1)

            inputs = self.ques_correct_to_hidden(torch.cat([raw_ques_embedding, corrects], dim=1))

            out, (h, c) = self.init_state_encoder(inputs)

            batch_init_h.append(h)
            batch_init_c.append(c)

        self.last_ques = torch.tensor(last_ques_list).to(self.device)

        self.last_n_ques = torch.stack(last_n_ques_list, dim=0)


        batch_init_h = torch.cat(batch_init_h, dim=0).unsqueeze(0)
        batch_init_c = torch.cat(batch_init_c, dim=0).unsqueeze(0)

        self.hc = (batch_init_h, batch_init_c)

        self.state_encoder_inputs = torch.zeros(batch_size, 1, self.hidden_dim*2).to(self.device)

        self.batch_state, self.hc = self.state_encoder(self.state_encoder_inputs, self.hc)

    def take_action(self):

        know_rep = self.know_representation.mean(dim=0)

        ques_representation = self.ques_representation + torch.mean(self.ques_representation, dim=0)
        batch_last_ques_representation = ques_representation[self.last_n_ques].mean(dim=1)  
        ques_representation = ques_representation.mean(dim=0).unsqueeze(0).expand(self.batch_size, -1)
        batch_last_ques_representation = ques_representation
        

        batch_state = self.batch_state.squeeze() 
        att_target = self_attention(batch_state.unsqueeze(1), self.batch_T_rep, self.batch_T_rep) 
        
        batch_know_rep_llm = self.W0(know_rep.unsqueeze(0).expand(self.batch_size, -1))
        batch_last_ques_representation_llm = self.W1(batch_last_ques_representation) 
        att_target_llm = self.W2(att_target.squeeze()) 
        batch_state_llm = self.W3(batch_state)  

        H_state = self.instruct_embeds_H + self.know_prompt_embeds + batch_know_rep_llm + self.targets_embeds + att_target_llm + self.student_embeds + batch_state_llm + self.response_embeds
        H_mask = torch.ones((self.batch_size, 1)).cuda(0)
        H_state = H_state.unsqueeze(1) 

        knowledges = self.llama_model_H(inputs_embeds=H_state.to(self.device), attention_mask=H_mask.to(self.device), return_dict=True)
        know_pooled_output = knowledges.last_hidden_state[:, -1]    
        know_pooled_logits = self.know_score(know_pooled_output) 

        know_prob = know_pooled_logits.softmax(dim=1)
 
        sampler = Categorical(probs=know_prob)
        while True:
            know = sampler.sample()
            if 0 not in know:
                break

        self.last_know_list.append(know)
        self.knowledge_prob_list.append(know_prob)  
        
   
        L_state = self.instruct_embeds_L + self.last_ques_embeds + batch_last_ques_representation_llm + self.targets_embeds + att_target_llm + self.student_embeds + batch_state_llm + self.response_embeds  
        L_mask = torch.ones((self.batch_size, 1)).cuda(0)
        L_state = L_state.unsqueeze(1)
        questions = self.llama_model_L(inputs_embeds=L_state.to(self.device), attention_mask=L_mask.to(self.device), return_dict=True)
        ques_pooled_output = questions.last_hidden_state[:, -1] 
        ques_pooled_logits = self.score(ques_pooled_output)
        ques_prob = ques_pooled_logits.softmax(dim=1)
        
        for i in range(self.batch_size):
            relevant_exercises = self.knowledge_to_questions[int(know[i])]
            relevant_exercises_set = set(relevant_exercises)
            irrelevant_exercises = set(range(self.ques_num)) - relevant_exercises_set
            ques_pooled_logits[i, list(irrelevant_exercises)] = float('-inf')
        
        sampler = Categorical(probs=ques_prob)
        action = sampler.sample()  

        self.last_ques_list.append(action)
        self.action_prob_list.append(ques_prob)
        
        return know, know_prob, action, ques_prob

    def step_refresh(self, ques_id, observation):
        ques_id = torch.tensor(ques_id, device=self.device)

        self.last_ques = ques_id
        try:
            if self.last_n_ques.size()[1] < self.n_ques:
                self.last_n_ques = torch.cat([self.last_n_ques, ques_id.unsqueeze(1).to(self.device)], dim=1)
            else:
                self.last_n_ques = torch.cat([self.last_n_ques[:, 1:], ques_id.unsqueeze(1).to(self.device)], dim=1)
        except AttributeError:
            if self.last_n_ques.size()[1] < 1:
                self.last_n_ques = torch.cat([self.last_n_ques, ques_id.unsqueeze(1).to(self.device)], dim=1)
            else:
                self.last_n_ques = torch.cat([self.last_n_ques[:, 1:], ques_id.unsqueeze(1).to(self.device)], dim=1)

        raw_ques_embedding = self.raw_ques_embedding[ques_id]

        corrects = torch.tensor(observation, device=self.device).view(-1, 1)

        inputs = self.ques_correct_to_hidden(torch.cat([raw_ques_embedding, corrects], dim=1))
        out, self.hc = self.init_state_encoder(inputs.unsqueeze(1), self.hc)

    def get_kt_prob(self, ques_id):
        ques_id = torch.tensor(ques_id, device=self.device)
        self.last_ques = ques_id
        prob = None

        self.state_encoder_inputs = self.ques_representation[ques_id].unsqueeze(1)

        self.batch_state, hc = self.state_encoder(self.state_encoder_inputs, self.hc)

        if self.use_kt:
            prob = self.sigmoid(self.kt_mlp(self.batch_state).squeeze())

        return prob

    def get_ranking_loss(self, seq): 
        return 
        
