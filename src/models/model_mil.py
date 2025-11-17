import torch
import torch.nn as nn
import torch.nn.functional as F

class MIL_fc(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., n_classes = 2, top_k=1,
                 embed_dim=1024):
        super().__init__()
        assert n_classes == 2
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifier=  nn.Linear(size[1], n_classes)
        self.top_k=top_k

    def forward(self, h, return_features=False):
        if h.dim() == 1:
            h = h.unsqueeze(0)
        h = self.fc(h)
        logits  = self.classifier(h) # K x 2
        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MIL_fc_mc(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., n_classes = 2, top_k=1, embed_dim=1024):
        super().__init__()
        assert n_classes > 2
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.top_k=top_k
        self.n_classes = n_classes
        assert self.top_k == 1
    
    def forward(self, h, return_features=False):    
        if h.dim() == 1:
            h = h.unsqueeze(0)
        h = self.fc(h) #768
        logits = self.classifiers(h)

        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]

        Y_hat = top_indices[1]
        Y_prob = y_probs[top_indices[0]]
        
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_indices[0])
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MIL_Mean(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., n_classes = 2, top_k=1, embed_dim=1024):
        super().__init__()
        assert n_classes >= 2
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.top_k=top_k
        self.n_classes = n_classes
        assert self.top_k == 1
    
    def forward(self, h, return_features=False):    
        if h.dim() == 1:
            h = h.unsqueeze(0)
        h = self.fc(h)  # 先通过全连接层
        h_mean = h.mean(dim=0, keepdim=True)  # 在n维上取平均，得到(1, d)
        logits = self.classifiers(h_mean)     # 通过分类器
        y_probs = F.softmax(logits, dim=1)
        # 对齐其他模型接口：第二个返回值应为按类概率分布（形状: 1 x n_classes）
        Y_hat = torch.argmax(y_probs, dim=1)
        results_dict = {}
        if return_features:
            results_dict.update({'features': h_mean})
        return logits, y_probs, Y_hat, y_probs, results_dict


class MIL_Max(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., n_classes = 2, top_k=1, embed_dim=1024):
        super().__init__()
        assert n_classes >= 2
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.top_k = top_k
        self.n_classes = n_classes
        assert self.top_k == 1

    def forward(self, h, return_features=False):
        if h.dim() == 1:
            h = h.unsqueeze(0)
        h = self.fc(h)
        # Max pooling over instance dimension
        h_max = torch.max(h, dim=0, keepdim=True)[0]
        logits = self.classifiers(h_max)
        y_probs = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(y_probs, dim=1)
        results_dict = {}
        if return_features:
            results_dict.update({'features': h_max})
        return logits, y_probs, Y_hat, y_probs, results_dict


class MIL_Attention(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., n_classes = 2, top_k=1,
                 embed_dim=1024, attention_dim: int = 128):
        super().__init__()
        assert n_classes >= 2
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.top_k = top_k
        self.n_classes = n_classes
        assert self.top_k == 1

        # Attention network: a = softmax(w^T tanh(Vh)) across instances
        self.attention_V = nn.Linear(size[1], attention_dim, bias=True)
        self.attention_w = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, h, return_features=False):
        if h.dim() == 1:
            h = h.unsqueeze(0)
        h = self.fc(h)  # K x d
        # Compute attention scores over K instances
        A = self.attention_w(torch.tanh(self.attention_V(h)))  # K x 1
        A = torch.softmax(A, dim=0)  # normalize over instances
        h_att = torch.sum(A * h, dim=0, keepdim=True)  # 1 x d
        logits = self.classifiers(h_att)
        y_probs = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(y_probs, dim=1)
        results_dict = {'attention_weights': A.squeeze(-1)}
        if return_features:
            results_dict.update({'features': h_att})
        return logits, y_probs, Y_hat, y_probs, results_dict
