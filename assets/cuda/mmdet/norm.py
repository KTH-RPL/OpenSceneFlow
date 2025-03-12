# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.registry import MODELS
from torch import distributed as dist
from torch import nn as nn
from torch.autograd.function import Function
import inspect
from typing import Dict, Tuple, Union
from mmengine.utils import is_tuple_of
from mmengine.utils.dl_utils.parrots_wrapper import (SyncBatchNorm, _BatchNorm,
                                                     _InstanceNorm)

MODELS.register_module('BN', module=nn.BatchNorm2d)
MODELS.register_module('BN1d', module=nn.BatchNorm1d)
MODELS.register_module('BN2d', module=nn.BatchNorm2d)
MODELS.register_module('BN3d', module=nn.BatchNorm3d)
MODELS.register_module('SyncBN', module=SyncBatchNorm)
MODELS.register_module('GN', module=nn.GroupNorm)
MODELS.register_module('LN', module=nn.LayerNorm)
MODELS.register_module('IN', module=nn.InstanceNorm2d)
MODELS.register_module('IN1d', module=nn.InstanceNorm1d)
MODELS.register_module('IN2d', module=nn.InstanceNorm2d)
MODELS.register_module('IN3d', module=nn.InstanceNorm3d)


def infer_abbr(class_type):
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'


def build_norm_layer(cfg: Dict,
                     num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    if inspect.isclass(layer_type):
        norm_layer = layer_type
    else:
        # Switch registry to the target scope. If `norm_layer` cannot be found
        # in the registry, fallback to search `norm_layer` in the
        # mmengine.MODELS.
        with MODELS.switch_scope_and_registry(None) as registry:
            norm_layer = registry.get(layer_type)
        if norm_layer is None:
            raise KeyError(f'Cannot find {norm_layer} in registry under '
                           f'scope name {registry.scope}')
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if norm_layer is not nn.GroupNorm:
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def is_norm(layer: nn.Module,
            exclude: Union[type, tuple, None] = None) -> bool:
    """Check if a layer is a normalization layer.

    Args:
        layer (nn.Module): The layer to be checked.
        exclude (type | tuple[type]): Types to be excluded.

    Returns:
        bool: Whether the layer is a norm layer.
    """
    if exclude is not None:
        if not isinstance(exclude, tuple):
            exclude = (exclude, )
        if not is_tuple_of(exclude, type):
            raise TypeError(
                f'"exclude" must be either None or type or a tuple of types, '
                f'but got {type(exclude)}: {exclude}')

    if exclude and isinstance(layer, exclude):
        return False

    all_norm_bases = (_BatchNorm, _InstanceNorm, nn.GroupNorm, nn.LayerNorm)
    return isinstance(layer, all_norm_bases)


class AllReduce(Function):

    @staticmethod
    def forward(ctx, input):
        input_list = [
            torch.zeros_like(input) for k in range(dist.get_world_size())
        ]
        # Use allgather instead of allreduce in-place operations is unreliable
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output



@MODELS.register_module('naiveSyncBN1d')
class NaiveSyncBatchNorm1d(nn.BatchNorm1d):
    """Syncronized Batch Normalization for 3D Tensors.

    Note:
        This implementation is modified from
        https://github.com/facebookresearch/detectron2/

        `torch.nn.SyncBatchNorm` has known unknown bugs.
        It produces significantly worse AP (and sometimes goes NaN)
        when the batch size on each worker is quite different
        (e.g., when scale augmentation is used).
        In 3D detection, different workers has points of different shapes,
        whish also cause instability.

        Use this implementation before `nn.SyncBatchNorm` is fixed.
        It is slower than `nn.SyncBatchNorm`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp16_enabled = False

    # customized normalization layer still needs this decorator
    # to force the input to be fp32 and the output to be fp16
    # TODO: make mmcv fp16 utils handle customized norm layers
    # @force_fp32(out_fp16=True)
    def forward(self, input):
        assert input.dtype == torch.float32, \
            f'input should be in float32 type, got {input.dtype}'
        if not dist.is_initialized() or dist.get_world_size() == 1 or not self.training:
            return super().forward(input)
        assert input.shape[0] > 0, 'SyncBN does not support empty inputs'
        dim_2 = input.dim() == 2
        if dim_2:
            input.unsqueeze_(2)

        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2])
        meansqr = torch.mean(input * input, dim=[0, 2])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (
            mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1)
        bias = bias.reshape(1, -1, 1)
        input = input * scale + bias
        if dim_2:
            input = input.squeeze(2)
        return input


# @NORM_LAYERS.register_module('naiveSyncBN2d')
class NaiveSyncBatchNorm2d(nn.BatchNorm2d):
    """Syncronized Batch Normalization for 4D Tensors.

    Note:
        This implementation is modified from
        https://github.com/facebookresearch/detectron2/

        `torch.nn.SyncBatchNorm` has known unknown bugs.
        It produces significantly worse AP (and sometimes goes NaN)
        when the batch size on each worker is quite different
        (e.g., when scale augmentation is used).
        This phenomenon also occurs when the multi-modality feature fusion
        modules of multi-modality detectors use SyncBN.

        Use this implementation before `nn.SyncBatchNorm` is fixed.
        It is slower than `nn.SyncBatchNorm`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp16_enabled = False

    # customized normalization layer still needs this decorator
    # to force the input to be fp32 and the output to be fp16
    # TODO: make mmcv fp16 utils handle customized norm layers
    # @force_fp32(out_fp16=True)
    def forward(self, input):
        assert input.dtype == torch.float32, \
            f'input should be in float32 type, got {input.dtype}'
        if not dist.is_initialized() or dist.get_world_size() == 1 or not self.training:
            return super().forward(input)
        # if dist.get_world_size() == 1 or not self.training:
        #     return super().forward(input)

        assert input.shape[0] > 0, 'SyncBN does not support empty inputs'
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (
            mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias

# @NORM_LAYERS.register_module('naiveSyncBN3d')
class NaiveSyncBatchNorm3d(nn.BatchNorm3d):
    """Syncronized Batch Normalization for 5D Tensors.

    Note:
        This implementation is modified from
        https://github.com/facebookresearch/detectron2/

        `torch.nn.SyncBatchNorm` has known unknown bugs.
        It produces significantly worse AP (and sometimes goes NaN)
        when the batch size on each worker is quite different
        (e.g., when scale augmentation is used).
        This phenomenon also occurs when the multi-modality feature fusion
        modules of multi-modality detectors use SyncBN.

        Use this implementation before `nn.SyncBatchNorm` is fixed.
        It is slower than `nn.SyncBatchNorm`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp16_enabled = False

    # customized normalization layer still needs this decorator
    # to force the input to be fp32 and the output to be fp16
    # TODO: make mmcv fp16 utils handle customized norm layers
    # @force_fp32(out_fp16=True)
    def forward(self, input):
        assert input.dtype == torch.float32, \
            f'input should be in float32 type, got {input.dtype}'
        if not dist.is_initialized() or dist.get_world_size() == 1 or not self.training:
            return super().forward(input)
        # if dist.get_world_size() == 1 or not self.training:
        #     return super().forward(input)

        assert input.shape[0] > 0, 'SyncBN does not support empty inputs'
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3, 4])
        meansqr = torch.mean(input * input, dim=[0, 2, 3, 4])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (
            mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1, 1)
        return input * scale + bias
