# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from pkg_resources import parse_version
from torch.utils.cpp_extension import load

if torch.cuda.is_available():
    try:
        import MultiScaleDeformableAttention as MSDA
    except ModuleNotFoundError as e:
        info_string = (
            "\n\nPlease compile MultiScaleDeformableAttention CUDA op with the following commands:\n"
            "\t`cd mask2former/modeling/pixel_decoder/ops`\n"
            "\t`sh make.sh`\n"
        )
        raise ModuleNotFoundError(info_string)
else:
    MultiScaleDeformableAttention = None


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()

def multi_scale_deformable_attention(
    value: Tensor, value_spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([height.item() * width.item() for height, width in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        sampling_value_l_ = grid_sample(value_l_, sampling_grid_l_)
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()



def grid_sample(input, grid):
    return _GridSample2dForward.apply(input, grid)

#----------------------------------------------------------------------------

def _should_use_custom_op():
    return enabled

#----------------------------------------------------------------------------

directory = '/segmatron/models/mask2former/modeling/pixel_decoder/ops/functions'
gridsample_grad2 = load(name='gridsample_grad2', sources=[f'{directory}/gridsample_cuda.cpp', f'{directory}/gridsample_cuda.cu'], verbose=True)

def grid_sample_2d(input, grid, padding_mode='zeros', align_corners=False):
    assert padding_mode in ['zeros', 'border']
    return _GridSample2dForward.apply(input, grid, padding_mode, align_corners)


def grid_sample_3d(input, grid, padding_mode='zeros', align_corners=False):
    assert padding_mode in ['zeros', 'border']
    return _GridSample3dForward.apply(input, grid, padding_mode, align_corners)


_use_pytorch_1_11_api = parse_version(torch.__version__) >= parse_version('1.11.0a')


class _GridSample2dForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, padding_mode="zeros", align_corners=False):
        assert input.ndim == 4
        assert grid.ndim == 4
        assert input.shape[0] == grid.shape[0]
        assert grid.shape[3] == 2

        output = torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear',
                                                 padding_mode=padding_mode, align_corners=align_corners)
        ctx.save_for_backward(input, grid)
        ctx.padding_mode = ['zeros', 'border'].index(padding_mode)
        ctx.align_corners = align_corners
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = _GridSample2dBackward.apply(grad_output, input, grid, ctx.padding_mode, ctx.align_corners)
        return grad_input, grad_grid, None, None

class _GridSample2dBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid, padding_mode="zeros", align_corners=False):
        op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')
        if _use_pytorch_1_11_api:
            output_mask = (ctx.needs_input_grad[1], ctx.needs_input_grad[2])
            grad_input, grad_grid = op(grad_output, input, grid, 0, padding_mode, align_corners, output_mask)
        else:
            grad_input, grad_grid = op(grad_output, input, grid, 0, padding_mode, align_corners)
        
        ctx.save_for_backward(grad_output, input, grid)
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
 
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        grad_output, input, grid = ctx.saved_tensors
        assert grad_output.is_cuda and input.is_cuda and grid.is_cuda and grad2_grad_input.is_cuda and grad2_grad_grid.is_cuda
        out = gridsample_grad2.grad2_2d(grad2_grad_input, grad2_grad_grid, grad_output,
                                        input, grid, ctx.padding_mode, ctx.align_corners)

        grad_grad_output = out[0]
        grad_input = out[1]
        grad_grid = out[2]
        
        return grad_grad_output, grad_input, grad_grid, None, None
