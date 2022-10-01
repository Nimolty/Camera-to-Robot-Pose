from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import warnings
import math
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import copy
from ..functions import MSDeformAttnFunction

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # self.value_proj = nn.Linear(d_model, d_model)
        for idx in range(self.n_levels// 2):
            if idx == 0:
                fc = nn.Linear(16, self.d_model)
                setattr(self, 'value_' + str(idx % (self.n_levels//2)), fc)
            elif idx == 1:
                fc = nn.Linear(32, self.d_model)
                setattr(self, 'value_' + str(idx % (self.n_levels//2)), fc)
            else:
                fc = nn.Linear(64, self.d_model)
                setattr(self, 'value_' + str(idx % (self.n_levels//2)), fc)
        
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        for idx in range(self.n_levels//2):
            fc = getattr(self, 'value_' + str(idx%(self.n_levels//2)))
            xavier_uniform_(fc.weight.data)
            constant_(fc.bias.data, 0.)

        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        # N, Len_in, _ = input_flatten.shape
        Len_in = (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() 

        value = []
        for idx, val in enumerate(input_flatten):
            fc = getattr(self, 'value_' + str(idx%(self.n_levels//2)))
            val = fc(val)
            value.append(val)
        value = torch.cat(value, dim=1)
        
        
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1,
                 n_levels=6, n_heads=8, n_points=6):
        super().__init__()
        
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # cross attention
        tgt2 = self.cross_attn(tgt,
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            output = layer(output, reference_points, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

        return output



class BaseModelCA(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None, d_ffn=1024, top_k=15):
        super(BaseModelCA, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.num_stacks = num_stacks
        self.heads = heads
        self.K = top_k
        
        # 创建decoder layer和decoder
        self.decoder_layer = DeformableTransformerDecoderLayer(opt.d_model, 
                                                          dropout=opt.dropout, 
                                                          n_levels=opt.levels, n_heads=opt.n_heads, n_points=opt.n_points)
        self.decoder = DeformableTransformerDecoder(self.decoder_layer, opt.num_decoder_layers)
        self.w_q = nn.Linear(2 * last_channel, last_channel)
        self.w_pos = nn.Linear(2, last_channel)
        self.pool = nn.MaxPool2d(kernel_size = (self.K, 1))
        for head in self.heads:
            classes = self.heads[head]
            head_conv = head_convs[head]
            if head != "hm":
                fc = nn.Sequential(nn.Linear(opt.d_model ,1024),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(1024, 1024),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(1024, 2))
                self.__setattr__(head, fc)
                continue
            if len(head_conv) > 0:
              out = nn.Conv2d(head_conv[-1], classes, 
                    kernel_size=1, stride=1, padding=0, bias=True)
              conv = nn.Conv2d(2 * last_channel, head_conv[0],
                               kernel_size=head_kernel, 
                               padding=head_kernel // 2, bias=True)
              convs = [conv]
              for k in range(1, len(head_conv)):
                  convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                               kernel_size=1, bias=True))
              if len(convs) == 1:
                fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
              elif len(convs) == 2:
                fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), out)
              elif len(convs) == 3:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), out)
              elif len(convs) == 4:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), 
                    convs[3], nn.ReLU(inplace=True), out)
              if 'hm' in head:
                fc[-1].bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(last_channel, classes, 
                  kernel_size=1, stride=1, padding=0, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)
        self._reset_parameters()
        
    
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                for param in m.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)

    def img2feats(self, x):
        raise NotImplementedError
    
    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
        raise NotImplementedError

    def forward(self, x, pre_img=None, pre_hm=None):
        if (pre_hm is not None) or (pre_img is not None):
            feats = self.imgpre2feats(x, pre_img, pre_hm)
        else:
            feats = self.img2feats(x)
        out = []
        for s in range(self.num_stacks):
            z = {}
            z["hm"] = self.__getattr__("hm")(feats[s])
            
            img_cur, img_pre = feats[1], feats[2] # 各自一个三元素的list
            hm_feature = z["hm"] #当前得到的heatmap proposal 为 Bx7xHxW
            B,C,H,W = hm_feature.shape
            topk_indices = torch.topk(hm_feature.view(B,C,-1), self.K, dim=-1)[1] # B x 7 x k
            # topk_indices = topk_indices.view(B, -1) # B x 105
            topk_locations = topk_indices.view(topk_indices.shape[0], topk_indices.shape[1], topk_indices.shape[2], \
                                               1, 1) # B x 7 x k x 1 x 1
            topk_locations = topk_locations.repeat(1, 1, 1, len(img_cur)+len(img_pre), 2)
            # 应该是 B x M x 6(表示6个scale) x 2 #(先x后y)
            topk_locations[:, :, :, :, 0] = topk_locations[:, :, :, :, 0] % W
            topk_locations[:, :, :, :, 1] = topk_locations[:, :, :, :, 1] // H
            
            # 保证在[0,1]之间，最后sigmoid
            topk_locations_sig = topk_locations.sigmoid()
            # B x 7 x k x n_levels x 2 
            topk_indices = topk_indices.view(B, -1)
            _, length = topk_indices.shape
            topk_ind1 = torch.arange(B).view(B, 1).repeat(1, length)
            
            query = feats[s].permute(0, 2, 3, 1).contiguous() # B x H x W x 2C
            B_q, _, _, C_q = query.shape
            topk_query = query.view(B_q, -1, C_q)[[topk_ind1, topk_indices]] # BxMx2C
            topk_query = topk_query.view(B_q, C, -1, C_q) # B x 7 x K x 2C
            
            topk_query = self.w_q(topk_query)
            topk_query = topk_query + self.w_pos(topk_locations[:, :, :, 0, :].float()) # B x 7 x K x C
             
            # 现在我们开始搭建decoder
            features_list = img_cur + img_pre
            src_flatten = []
            spatial_shapes = []
            for src in features_list:
                b, c, h, w = src.shape
                src = src.flatten(2).transpose(1, 2)
                src_flatten.append(src)
                spatial_shapes.append((h, w))
            # src_flatten = torch.cat(src_flatten, dim=1)
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
             
            output = []
            for c in range(C):
                out_ = self.decoder(topk_query[:, c, :, :], topk_locations_sig[:, c, :, :, :], src_flatten, spatial_shapes, level_start_index)
                out_ = self.pool(out_)
                output.append(out_)
            output = torch.cat(output,dim=1)
            # print('output', output.shape)
            # print('output.shape', output.shape)
            # output为B x M x C
            for head in self.heads:
                if head != "hm":
                    z[head] = self.__getattr__(head)(output)
            out.append(z)  
        
        return out