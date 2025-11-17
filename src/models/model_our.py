# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from os.path import join as pjoin
# from .model_utils import *
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
# from trident.slide_encoder_models.load import slide_to_patch_encoder_name
from transformers import AutoModel
# from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

import logging
from transformers import CLIPTextModel, CLIPTokenizer
from models.load import encoder_factory


class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x, y=None):
        if y is None:
            y = x
        a = self.attention_a(x)
        b = self.attention_b(y)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x



def clip_gradients(model, max_norm=1.0):
    """Clipping gradients"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def safe_softmax(x, dim=-1, eps=1e-8):
    """Numerically stable softmax"""
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    return x_exp / (torch.sum(x_exp, dim=dim, keepdim=True) + eps)

def monitor_gradients(model, step):
    """Monitor gradient norm"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    if step % 100 == 0:
        print(f"Step {step}, Gradient norm: {total_norm:.4f}")
    
    return total_norm

class ClinicalTextEncoder(nn.Module):
    """Clinical text encoder using CLIP pretrained text encoder"""
    def __init__(self, clip_path="/data/zzh/WSI_zzh/CLAM/clip-vit-large-patch14.bin", embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        
        # print(f"Attempting to load CLIP model from: {clip_path}")
        
        import os
        if not os.path.exists(clip_path):
            print(f"Error: CLIP model file not found at {clip_path}")
            raise FileNotFoundError(f"CLIP model file not found at {clip_path}")
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14",
            local_files_only=False,
            torch_dtype=torch.float16
        )

        self.text_model = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            local_files_only=False,
            torch_dtype=torch.float16
        )
            
        if self.text_model is not None:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        if self.text_model is not None:
            self.output_projection = nn.Sequential(
                nn.Linear(self.text_model.config.hidden_size, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            for param in self.output_projection.parameters():
                param.requires_grad = False
        else:
            self.output_projection = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            )
        
        self.gradient_scale = 1.0 / (embed_dim ** 0.5)
        
    def forward(self, clinical_texts):
        """
        Args:
            clinical_texts: List[str]
        Returns:
            torch.Tensor: text features [batch_size, embed_dim]
        """
        batch_size = len(clinical_texts)
        device = next(self.parameters()).device
        
        tokens = self.tokenizer(
            clinical_texts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        
        with torch.no_grad():
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        text_features = outputs.last_hidden_state
        
        text_features = text_features.to(torch.float32)
        
        if torch.isnan(text_features).any() or torch.isinf(text_features).any():
            print("Warning: NaN or Inf detected in CLIP text_features")
            text_features = torch.nan_to_num(text_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        text_features = text_features.mean(dim=1)
        
        text_features = self.output_projection(text_features) * self.gradient_scale
        
        if torch.isnan(text_features).any() or torch.isinf(text_features).any():
            print("Warning: NaN or Inf detected in final text_features")
            text_features = torch.nan_to_num(text_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return text_features


class ELSiP(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.size_dict = {"small": [self.embed_dim, 512, 256], "big": [self.embed_dim, 1024, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        
        self.feature_encoder = nn.Sequential(*fc)
        self.attention_net = attention_net
        self.Cross_attention = CrossAttentionEnhancer(L=self.embed_dim, D=self.embed_dim, dropout=dropout)

        fc2 = [nn.Linear(embed_dim, size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc2 = nn.Sequential(*fc2)
        self.classifiers = nn.Linear(self.embed_dim * 3, n_classes)
        # self.classifiers3 = nn.Linear(self.embed_dim * 3, n_classes)

        instance_classifiers = [nn.Linear(size[0], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        self.clinical_text_encoder = ClinicalTextEncoder(embed_dim=self.embed_dim)
        self.patch_encoder = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        # self.patch_encoder = encoder_factory('madeleine')

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, s_f, p_f, clinical_text, patch_coords=None, patch_size_lv0=None, label=None, instance_eval=False, enhance=False, return_features=False, attention_only=False):
        device = s_f.device
        if s_f.dim() == 1:
            s_f = s_f.unsqueeze(0)
        results_dict = {}

        text_features = self.clinical_text_encoder(clinical_text)

        attention_scores, p_f_enhanced = self.Cross_attention(p_f, s_f)
        if attention_only:
            return attention_scores
        A_raw = attention_scores
        attention_scores = F.softmax(attention_scores, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    instance_loss, preds, targets = self.inst_eval(attention_scores, p_f_enhanced, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(attention_scores, p_f_enhanced, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        self.patch_encoder = self.patch_encoder.to(device)
        with torch.inference_mode():
            s_f_enhanced = self.patch_encoder.encode_slide_from_patch_features(p_f_enhanced, patch_coords, patch_size_lv0)
        
        M = torch.cat([s_f_enhanced, s_f, text_features], dim=1)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': s_f_enhanced})
        return logits, Y_prob, Y_hat, A_raw, results_dict


class CrossAttentionEnhancer(nn.Module):
    """
    Cross attention module to enhance x using y.
    x: [n, L] - features to be enhanced
    y: [L] - query feature for enhancement
    A: [n, L] - output dimension same as x
    """
    def __init__(self, L=768, D=768, dropout=False):
        super().__init__()
        self.L = L
        self.D = D
        
        self.query_proj = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh()
        )
        
        self.key_proj = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh()
        )
        
        self.value_proj = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh()
        )
        
        self.gate_proj = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Linear(D, L)
        
        self.residual_weight = nn.Parameter(torch.ones(1))
        
        if dropout:
            self.dropout = nn.Dropout(0.25)
        else:
            self.dropout = None
            
        self.scale = D ** 0.5
        
    def forward(self, x, y):
        """
        Args:
            x: [n, L] - features to be enhanced
            y: [L] - query feature for enhancement
        Returns:
            attention_scores: [1, n]
            enhanced_x: [n, L]
        """
        n = x.size(0)
        
        if y.dim() == 1:
            y = y.unsqueeze(0)
        
        query = self.query_proj(y)
        keys = self.key_proj(x)
        values = self.value_proj(x)
        
        attention_scores = torch.mm(query, keys.t()) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attended_values = torch.mm(attention_weights, values)
        
        gate = self.gate_proj(x)
        
        attended_values = attended_values.expand(n, -1)
        
        gated_values = attended_values.mul(gate)
        
        enhanced_features = self.output_proj(gated_values)
        
        enhanced_x = x + self.residual_weight * enhanced_features
        
        if self.dropout:
            enhanced_x = self.dropout(enhanced_x)
            
        return attention_scores, enhanced_x


