import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim, pos_dim, num_heads=8, num_patches=197, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.pos_dim = pos_dim
        self.num_patches = num_patches
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.pos_emb = nn.Parameter(torch.zeros(num_patches, num_patches, pos_dim))
        self.pos_proj  = nn.Linear(pos_dim, num_heads, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_att_scores=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) # B, H, N, C/H

        pos_score = self.pos_proj(self.pos_emb).permute(2,0,1) # H, N, N

        attn = ((q @ k.transpose(-2, -1)) + pos_score) * self.scale # B, H, N, N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_att_scores :
            return attn.mean(0)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # B, H, N, C/H
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def local_init(self):

        self.qkv.weight[:self.dim * 2].data.fill_(0)
        img_size = int(self.num_patches**.5)

        for i in range(self.pos_dim//2):
            ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
            indx = ind.repeat(img_size,img_size)
            indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)
            indd = indx**2 + indy**2
            self.pos_emb[:,:,0] = indx
            self.pos_emb[:,:,1] = indy
            self.pos_emb[:,:,2] = indd
            self.pos_emb[:,:,3:].fill_(0)

        kernel_size = int(self.num_heads**.5)
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1+kernel_size*h2
                self.pos_proj.weight.data[position,0] = 2*(h2-kernel_size//2)
                self.pos_proj.weight.data[position,1] = 2*(h1-kernel_size//2)
                self.pos_proj.weight.data[position,2] = -1

def test_indices():
    num_patches = 4
    ind = torch.arange(num_patches**.5).view(1,-1) - torch.arange(num_patches**.5).view(-1, 1)#+num_patches-1
    indx = ind.repeat((int(num_patches**.5),int(num_patches**.5)))
    indy = ind.repeat_interleave(int(num_patches**.5),dim=0).repeat_interleave(int(num_patches**.5),dim=1)
    indd = indx**2 + indy**2
    print(indx)
    print(indy)
    print(indd)

def test_attention():
    num_heads = 9
    num_patches = 16
    dim = 10*num_heads
    img_size = int(num_patches**.5)
    att = Attention(dim=dim,pos_dim=4,num_heads=num_heads,num_patches=num_patches)
    att.local_init()

    x = torch.randn(1, num_patches, dim)
    out = att(x, return_att_scores = True).detach()
    # fig, axarr = plt.subplots(1,num_heads, figsize=(num_heads*4,4))
    # for h in range(num_heads):
    #     axarr[h].matshow(out[h,9].view(img_size, img_size))
    #     axarr[h].set_title('Head %d'%h)
    # plt.tight_layout()


if __name__ == '__main__':
    test_attention()