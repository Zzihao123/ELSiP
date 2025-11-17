import torch
import timm
import os

class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50.tv_in1k', 
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0}, 
                 pool: bool = True,
                 local_weights_path: str = None):
        super().__init__()
        
        # 如果提供了本地权重路径，则不使用pretrained=True
        if local_weights_path and os.path.exists(local_weights_path):
            print(f"Loading local weights from: {local_weights_path}")
            kwargs['pretrained'] = False
            self.model = timm.create_model(model_name, **kwargs)
            # 加载本地权重
            state_dict = torch.load(local_weights_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
        else:
            assert kwargs.get('pretrained', False), 'only pretrained models are supported'
            self.model = timm.create_model(model_name, **kwargs)
            
        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
    
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out