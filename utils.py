import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import pickle
from collections import OrderedDict


class Project(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Project, self).__init__()
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),  
            nn.ReLU(),                      
            nn.Linear(128, self.output_dim)  
        )
        
    def forward(self, x):
        return self.mlp(x)


with open('./data/assist_problem_skills_relation.pkl', 'rb') as file:
    assist_data = pickle.load(file)

with open('./data/junyi_problem_skills_relation.pkl', 'rb') as file:
    junyi_data = pickle.load(file)




def build_knowledge_to_questions_mapping(type, data):
    knowledge_to_questions = {}
    if type == 'assist':
        for question_id, knowledge_ids in data.items():
            for knowledge_id in knowledge_ids:
                if knowledge_id == 0:
                    continue
                if knowledge_id in knowledge_to_questions:
                    knowledge_to_questions[knowledge_id].append(question_id)
                else:
                    knowledge_to_questions[knowledge_id] = [question_id]
    elif type == 'junyi':
        for question_id, knowledge_ids in data.items():
            for knowledge_id in knowledge_ids:
                if knowledge_id not in knowledge_to_questions:
                    knowledge_to_questions[knowledge_id] = []
                knowledge_to_questions[knowledge_id].append(question_id)

    return knowledge_to_questions


def find_related_questions_for_list(question_ids_list, question_to_knowledge, knowledge_to_questions):
    related_questions_collection = {}  
    all_candidate_questions = set()  

    for question_id in question_ids_list:
        knowledge_ids = question_to_knowledge.get(question_id, [])
       

        candidate_questions = set()
        for knowledge_id in knowledge_ids:
            if knowledge_id != 0 and knowledge_id in knowledge_to_questions:
                candidate_questions.update(knowledge_to_questions[knowledge_id])

        related_questions_collection[question_id] = candidate_questions
        all_candidate_questions.update(candidate_questions)

    all_candidate_questions_list = list(all_candidate_questions)
    return related_questions_collection, all_candidate_questions_list



class MLP(nn.Module):
    def __init__(self, hidden_list):
        super(MLP, self).__init__()

        self.model = nn.Sequential(OrderedDict([('layer{}'.format(i), nn.Linear(hidden_list[i], hidden_list[i+1]))
                                                for i in range(len(hidden_list) - 1)]))


# https://zhuanlan.zhihu.com/p/127030939
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size()  

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)  # batch_size * n_head * n * n

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. pass them softmax to make [0, 1] range
       
        score = F.softmax(score, dim=3)

        # 4. multiply with Value
        v = score @ v  # batch_size * n_head * n * head_dim

        return v, score


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, n_head, dropout_rate):  
        super(MultiHeadAttention, self).__init__()

        self.model_dim = model_dim
        self.n_head = n_head
        self.head_dim = self.model_dim  

        self.linear_k = nn.Linear(self.model_dim, self.head_dim * self.n_head)
        self.linear_v = nn.Linear(self.model_dim, self.head_dim * self.n_head)
        self.linear_q = nn.Linear(self.model_dim, self.head_dim * self.n_head)

        self.linear_final = nn.Linear(self.head_dim * self.n_head, self.model_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.scaled_dot_product_attention = ScaleDotProductAttention()

    def forward(self, inputs, mask=None):  
        q = self.linear_q(inputs)
        k = self.linear_k(inputs)
        v = self.linear_v(inputs)
        batch_size = k.size()[0]

        q_ = q.view(batch_size, self.n_head, -1, self.head_dim)  
        k_ = k.view(batch_size, self.n_head, -1, self.head_dim)
        v_ = v.view(batch_size, self.n_head, -1, self.head_dim)

        context, _ = self.scaled_dot_product_attention(q_, k_, v_, mask)  # context: batch_size * n_head * n * head_dim
        # output: batch_size * n_head * n * head_dim
        #      => batch_size * n * n_head * head_dim
        #      => batch_size * n * (n_head * head_dim)
        output = context.transpose(1, 2) .contiguous().view(batch_size, -1, self.n_head * self.head_dim)
        output = self.linear_final(output)  # => batch_size * n * model_dim
        output = self.dropout(output)
        return output


class FeedForward(nn.Module):

    def __init__(self, model_dim, hidden, drop_prob=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(model_dim, hidden)
        self.linear2 = nn.Linear(hidden, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, hidden_dim, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim=hidden_dim, n_head=n_head, dropout_rate=drop_prob)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = FeedForward(model_dim=hidden_dim, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, s_mask):
        _x = x  
        x = self.attention(x, mask=s_mask)  
        x = self.norm1(x + _x)  
        x = self.dropout1(x)

        _x = x  
        x = self.ffn(x)  
        x = self.norm2(x + _x) 
        x = self.dropout2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, ffn_hidden, n_head, n_layers, drop_prob):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(hidden_dim=input_dim,
                                                  ffn_hidden=ffn_hidden,  
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, s_mask=None):

        for layer in self.layers:
            x = layer(x, s_mask)

        return x

# q: batch_size*n*q_input_dim. k: batch_size*item_num*k_input_dim. v:batch_size*item_num*v_input_dim
def self_attention(q, k, v, hidden_dim=None, value_dim=None):
    q_input_dim = q.shape[-1]
    k_input_dim = k.shape[-1]
    v_input_dim = v.shape[-1]
    if not hidden_dim:
        hidden_dim = q_input_dim
    if not value_dim:
        value_dim = v_input_dim

    device = q.device
    W1 = nn.Linear(q_input_dim, hidden_dim, device=device)
    W2 = nn.Linear(k_input_dim, hidden_dim, device=device)
    W3 = nn.Linear(v_input_dim, value_dim, device=device)

    score = torch.matmul(W1(q), W2(k).transpose(1, 2)) / math.sqrt(hidden_dim)  # batch_size*n*item_num
    score = F.softmax(score, dim=2)  # batch_size*n*item_num
    v = torch.matmul(score, W3(v))  # batch_size*n*value_dim
    return v
