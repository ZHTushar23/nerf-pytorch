import torch
import torch.nn as nn
import torch.nn.functional as F
from data_preprocess import *
class CloudNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=60, output_ch=4, skips=[4], use_viewdirs=False):
        super(CloudNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
    
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)


    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        print(input_pts.shape)
        print(input_views.shape)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    
    
if __name__=="__main__":
    model = CloudNeRF(input_ch=60,input_ch_views=24,use_viewdirs=True)
    x = torch.rand(10,3)
    d = torch.rand(10,3)

    embed_fn1, input_ch1 = get_embedder(10)
    xp = embed_fn1(x)

    embed_fn2, input_ch2 = get_embedder(4)
    dp = embed_fn2(d)
    inputs = torch.cat([xp, dp], -1)

    out = model(inputs)
    print(out.shape)
    print("Done!")    