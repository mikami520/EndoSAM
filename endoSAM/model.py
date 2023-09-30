'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2023-09-16 19:47:31
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2023-09-30 16:56:23
FilePath: /EndoSAM/endoSAM/model.py
Description: EndoSAM model adapter 
I Love IU
Copyright (c) 2023 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from einops import rearrange
from utils import postprocess_masks

class EndoSAMAdapter(nn.Module):
    def __init__(self, device, 
                 num_classes,
                 sam_mask_encoder,
                 sam_prompt_encoder,
                 sam_mask_decoder,
                 num_token=8,
                 ):
        super(EndoSAMAdapter, self).__init__()
        self.device = device
        self.num_classes = num_classes - 1
        self.num_token = num_token
        self.sam_mask_encoder = sam_mask_encoder.to(self.device)
        self.sam_prompt_encoder = sam_prompt_encoder.to(self.device)
        self.sam_mask_decoder = sam_mask_decoder.to(self.device)
        self.prototype_prompt_encoder = Prototype_Prompt_Encoder(feat_dim=256, 
                                                            hidden_dim_dense=128, 
                                                            hidden_dim_sparse=128, 
                                                            size=64, 
                                                            num_tokens=self.num_token).to(self.device)
        self.learnable_prototypes_model = Learnable_Prototypes(num_classes=self.num_classes, feat_dim = 256).to(self.device)
        self.prototypes = self.learnable_prototypes_model()
        self.sam_mask_encoder.to(self.device)
        self.sam_prompt_encoder.to(self.device)
        self.sam_mask_decoder.to(self.device)
        
        for _, param in self.prototype_prompt_encoder.named_parameters():
            param.requires_grad = True
        
        for _, param in self.learnable_prototypes_model.named_parameters():
            param.requires_grad = True
        
        for _, param in self.sam_mask_decoder.named_parameters():
            param.requires_grad = True
        
        for _, param in self.sam_mask_encoder.named_parameters():
            param.requires_grad = False
        
        for _, param in self.sam_prompt_encoder.named_parameters():
            param.requires_grad = False


    def forward(self, x):
        sam_features = self.sam_mask_encoder(x)
        sam_features = rearrange(sam_features, 'b c h w -> b (h w) c')
        cls_ids = torch.tensor(1).repeat(sam_features.shape[0]).to(self.device)
        dense_embeddings, sparse_embeddings = self.prototype_prompt_encoder(sam_features, self.prototypes, cls_ids, self.num_classes)
        pred = []
        pred_quality = []
        sam_features = rearrange(sam_features,'b (h w) c -> b c h w', h=64, w=64)
        for dense_embedding, sparse_embedding, features_per_image in zip(dense_embeddings.unsqueeze(1), sparse_embeddings.unsqueeze(1), sam_features):    
            low_res_masks_per_image, mask_quality_per_image = self.sam_mask_decoder(
                    image_embeddings=features_per_image.unsqueeze(0),
                    image_pe=self.sam_prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=sparse_embedding,
                    dense_prompt_embeddings=dense_embedding, 
                    multimask_output=True,
                )
            pred_per_image = postprocess_masks(
                low_res_masks_per_image,
                input_size=(819, 1024),
                original_size=(1024, 1280),
            )
            
            pred.append(pred_per_image)
            pred_quality.append(mask_quality_per_image)
        
        pred = torch.cat(pred, dim=0)
        pred_quality = torch.cat(pred_quality,dim=0)
        return pred, pred_quality

class Prototype_Prompt_Encoder(nn.Module):
    def __init__(self, feat_dim=256, 
                 hidden_dim_dense=128, 
                 hidden_dim_sparse=128, 
                 size=64, 
                 num_tokens=8):
                
        super(Prototype_Prompt_Encoder, self).__init__()
        self.dense_fc_1 = nn.Conv2d(feat_dim, hidden_dim_dense, 1)
        self.dense_fc_2 = nn.Conv2d(hidden_dim_dense, feat_dim, 1)
        
        self.relu = nn.ReLU()

        self.sparse_fc_1 = nn.Conv1d(size*size, hidden_dim_sparse, 1)
        self.sparse_fc_2 = nn.Conv1d(hidden_dim_sparse, num_tokens, 1)
        
        
        pn_cls_embeddings = [nn.Embedding(num_tokens, feat_dim) for _ in range(2)] # one for positive and one for negative 

            
        self.pn_cls_embeddings = nn.ModuleList(pn_cls_embeddings)
                
    def forward(self, feat, prototypes, cls_ids, num_classes):
  
        cls_prompts = prototypes.unsqueeze(-1)
        cls_prompts = torch.stack([cls_prompts for _ in range(feat.size(0))], dim=0)

        
        feat = torch.stack([feat for _ in range(cls_prompts.size(1))], dim=1)
        # compute similarity matrix 
        sim = torch.matmul(feat, cls_prompts)
        
        # compute class-activated feature
        feat =  feat + feat*sim
        feat_sparse = feat.clone()
        
        # compute dense embeddings
        one_hot = torch.nn.functional.one_hot(cls_ids-1,num_classes)
        feat = feat[one_hot == 1]
        feat = rearrange(feat.squeeze(1),'b (h w) c -> b c h w', h=64, w=64)
        dense_embeddings = self.dense_fc_2(self.relu(self.dense_fc_1(feat)))
        
        # compute sparse embeddings
        feat_sparse = rearrange(feat_sparse,'b num_cls hw c -> (b num_cls) hw c')
        sparse_embeddings = self.sparse_fc_2(self.relu(self.sparse_fc_1(feat_sparse)))
        sparse_embeddings = rearrange(sparse_embeddings,'(b num_cls) n c -> b num_cls n c', num_cls=1)
        
        pos_embed = self.pn_cls_embeddings[1].weight.unsqueeze(0).unsqueeze(0) * one_hot.unsqueeze(-1).unsqueeze(-1)
        neg_embed = self.pn_cls_embeddings[0].weight.unsqueeze(0).unsqueeze(0) * (1-one_hot).unsqueeze(-1).unsqueeze(-1)
        

        sparse_embeddings = sparse_embeddings + pos_embed.detach() + neg_embed.detach()
            
        sparse_embeddings = rearrange(sparse_embeddings,'b num_cls n c -> b (num_cls n) c')
        
        return dense_embeddings, sparse_embeddings 


class Learnable_Prototypes(nn.Module):
    def __init__(self, num_classes=7 , feat_dim=256):
        super(Learnable_Prototypes, self).__init__()
        self.class_embeddings = nn.Embedding(num_classes, feat_dim)
        
    def forward(self):
        return self.class_embeddings.weight