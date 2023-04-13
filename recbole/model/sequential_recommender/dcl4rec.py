# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import torch.nn.functional as F

import pickle

class DCL4Rec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(DCL4Rec, self).__init__(config, dataset)

        # load parameters info
        self.config = config
        self.lambda_cts = config['LAMBDA_CTS']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.K = config['K']
        self.temp = config['TEMP']
        self.phi = config['PHI']
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0) #item*hidden_size
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        #where is the max_seq_length
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.noise =  torch.randn([self.K,self.hidden_size])
        self.noise.requires_grad = True
        
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.batch_size = config['train_batch_size']
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.CrossEntropy = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1] #batch * item_seq * hidden_size
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]
    
    def device_as(self, t1, t2):
        """
        Moves t1 to the device of t2
        """
        return t1.to(t2.device)

    def sim(self,x,y):
        cos = nn.CosineSimilarity(dim=-1)
        return cos(x,y)/self.temp

    def cts_loss2(self, proj_1, proj_2):
        # mask = (~torch.eye(proj_1.shape[0] * 2, proj_1.shape[0] * 2, dtype=bool)).float()
        batch_size = proj_1.shape[0]
        N = batch_size + self.K

        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)
        
        
        representations = torch.cat([z_i,self.device_as(z_j,z_i)],dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)


        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.K)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temp)

        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(self.K):
            if similarity_matrix[i,batch_size+i] > self.phi:
                mask[i, batch_size + i] = 0
                mask[batch_size + i, i] = 0


        denominator =  self.device_as(mask, similarity_matrix) * torch.exp(similarity_matrix / self.temp)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / N
        return loss

    def cts_loss(self, z_i, z_j, temp, batch_size):  # B * D    B * D
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)  # 2B * D

        sim = torch.mm(z, z.T) / temp  # 2B * 2B

        sim_i_j = torch.diag(sim, batch_size)  # B*1
        sim_j_i = torch.diag(sim, -batch_size)  # B*1

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N * C
        return logits, labels

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask


    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len) #batch*hidden_size
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)

        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) #batch*hidden_size @ hidden_size*num_item
            loss = self.loss_fct(logits, pos_items)

        raw_seq_output = self.forward(item_seq, item_seq_len)
        

        
        if self.config['aug'] == 'self':
            cts_seq_output = self.forward(item_seq, item_seq_len)
        else:
            cts_aug, cts_aug_lengths = interaction['aug'], interaction['aug_lengths']
            cts_seq_output = self.forward(cts_aug, cts_aug_lengths)

        cts_nce_logits, cts_nce_labels = self.cts_loss(raw_seq_output, cts_seq_output, temp=1.0,
                                                       batch_size=item_seq_len.shape[0])
        nce_loss1 = self.loss_fct(cts_nce_logits, cts_nce_labels)
        
        
        self.noise = self.noise.to(cts_seq_output.device)

        
        cos_sim = self.sim(raw_seq_output.unsqueeze(1),cts_seq_output.unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long().to(cts_seq_output.device)
        los_fnt = nn.CrossEntropyLoss()
        
        # for _ in range(4):
        #     cos_sim_negative = self.sim(raw_seq_output.unsqueeze(1), self.noise.unsqueeze(0))
        #     cos_sim_fused = torch.cat([cos_sim, cos_sim_negative], 1)
        #     loss1 = los_fnt(cos_sim_fused,labels)
        #     noise_grad = torch.autograd.grad(loss1,self.noise,retain_graph=True)[0]
        #     self.noise = self.noise + (noise_grad/torch.norm(noise_grad,dim=-1,keepdim=True)).mul_(1e-3)
        #     self.noise = torch.where(torch.isnan(self.noise),torch.zeros_like(self.noise),self.noise)
        
        cos_sim_negative = self.sim(raw_seq_output.unsqueeze(1),self.noise.unsqueeze(0))
        cos_sim = torch.cat([cos_sim,cos_sim_negative],1)
        label_dis = torch.cat([torch.eye(cos_sim.size(0),device = cos_sim.device)[labels],torch.zeros_like(cos_sim_negative)],-1)
        weights = torch.where(cos_sim>self.phi,0,1)
        mask_weights = torch.eye(cos_sim.size(0),device = cos_sim.device) - torch.diag_embed(torch.diag(weights))
        weights = weights + torch.cat([mask_weights,torch.zeros_like(cos_sim_negative)],-1)
        
        soft_cos_sim = torch.softmax(cos_sim*weights,-1)
        loss2 = - (label_dis*torch.log(soft_cos_sim))+(1-label_dis)*torch.log(1-soft_cos_sim)
        loss2 = torch.mean(loss2)

        return loss + self.lambda_cts*nce_loss1 + self.lambda_cts*loss2


    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
    
    def full_sort_predict_test(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        with open('/exp1/embedding/DCL4Rec_book.pkl', 'wb') as handle:
            pickle.dump(test_items_emb,handle)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
