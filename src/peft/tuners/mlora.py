import math
import re
import warnings
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    CLAMP_QUANTILE,
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_MLORA_TARGET_MODULES_MAPPING,
    ACTIVATION_FUNCTION_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class MLoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`MLoraModel`].

    Args:
        r (`List[int]`): MLora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply MLora to.
        mlora_alpha (`int`): The alpha parameter for MLora scaling.
        mlora_dropout (`float`): The dropout probability for MLora layers.
        mlora_af (`str`): The activation function for MLora layers.
        mlora_af_sin_A (`float`): The A parameter for the Sin function used in MLora layers (Only works when mlora_af is "Sin").
        mlora_af_sin_omega (`float`): The omega parameter for the Sin function used in MLora layers (Only works when mlora_af is "Sin").
        mlora_use_P (`bool`): Set this to True if Ps are used in MLora.
        mlora_use_b (`bool`): Set this to True if bs are used in MLora.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for MLora. Can be 'none', 'all' or 'mlora_only'
        modules_to_save (`List[str]`):List of modules apart from MLoRA layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the MLoRA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the MLoRA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    r: List[int] = field(default=None, metadata={"help": "MLora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with MLora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    mlora_alpha: int = field(default=8, metadata={"help": "MLora alpha"})
    mlora_dropout: float = field(default=0.0, metadata={"help": "MLora dropout"})
    mlora_af: str = field(default="Sigmoid", metadata={"help": "Activation function used in Mlora layers"})
    mlora_af_sin_A: float = field(default=0.0005,
                                  metadata={"help": "A parameter for the Sin function used in MLora layers"})
    mlora_af_sin_omega: float = field(default=1000,
                                      metadata={"help": "Omega parameter for the Sin function used in MLora layers"})
    mlora_use_P: bool = field(default=True, metadata={"help": "Set this to True if Ps are used in MLora"})
    mlora_use_b: bool = field(default=True, metadata={"help": "Set this to True if bs are used in MLora"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for MLora. Can be 'none', 'all' or 'mlora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from MLoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_mlora_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the MLora layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[List, int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.MLORA


class MLoraModel(torch.nn.Module):
    """
    Creates Multi-layer Low Rank Adapter (MLora) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`MLoraConfig`]): The configuration of the MLora model.

    Returns:
        `torch.nn.Module`: The MLora model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM, MLoraConfig
        >>> from peft import MLoraModel, MLoraConfig

        >>> config = MLoraConfig(
        ...     peft_type="MLORA",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=[2, 4, 8],
        ...     mlora_alpha=32,
        ...     target_modules=["q", "v"],
        ...     mlora_dropout=0.01,
        ...     mlora_af="Sigmoid",
        ...     mlora_use_P=True,
        ...     mlora_use_b=True,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> mlora_model = MLoraModel(config, model)
        ```

        ```py
        >>> import transformers
        >>> from peft import MLoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = MLoraConfig(
        ...     r=[2, 4, 8], mlora_alpha=16, target_modules=target_modules, mlora_dropout=0.1, mlora_af="Sigmoid", mlora_use_P=True, mlora_use_b=True, bias="none", task_type="CAUSAL_LM"
        ... )

        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     load_in_8bit=True,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> mlora_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`MLoraConfig`]): The configuration of the MLora model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

        # transformers models have a .config attribute, whose presence is assumed later on
        if not hasattr(self, "config"):
            self.config = {"model_type": "custom"}

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = getattr(self.model, "config", {"model_type": "custom"})
            if hasattr(model_config, "to_dict"):
                model_config = model_config.to_dict()

            config = self._prepare_mlora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "MLoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_mlora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available():
            raise ImportError(
                "To use MLora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

    def _check_target_module_exists(self, mlora_config, key):
        if isinstance(mlora_config.target_modules, str):
            target_module_found = re.fullmatch(mlora_config.target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in mlora_config.target_modules)
            is_using_layer_indexes = getattr(mlora_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(mlora_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(mlora_config.layers_to_transform, int):
                            target_module_found = layer_index == mlora_config.layers_to_transform
                        else:
                            target_module_found = layer_index in mlora_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found

    def _create_new_module(self, mlora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": mlora_config.r,
            "mlora_alpha": mlora_config.mlora_alpha,
            "mlora_dropout": mlora_config.mlora_dropout,
            "mlora_af": mlora_config.mlora_af,
            "mlora_af_sin_A": mlora_config.mlora_af_sin_A,
            "mlora_af_sin_omega": mlora_config.mlora_af_sin_omega,
            "mlora_use_P": mlora_config.mlora_use_P,
            "mlora_use_b": mlora_config.mlora_use_b,
            "fan_in_fan_out": mlora_config.fan_in_fan_out,
            "init_mlora_weights": mlora_config.init_mlora_weights,
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = mlora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = mlora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)

        return new_module

    def _find_and_replace(self, adapter_name):
        mlora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            if not self._check_target_module_exists(mlora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, MLoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    mlora_config.r,
                    mlora_config.mlora_alpha,
                    mlora_config.mlora_dropout,
                    mlora_config.mlora_af,
                    mlora_config.mlora_af_sin_A,
                    mlora_config.mlora_af_sin_omega,
                    mlora_config.mlora_use_P,
                    mlora_config.mlora_use_b,
                    mlora_config.init_mlora_weights,
                )
            elif isinstance(target, MLoraLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer_embedding(
                    adapter_name,
                    mlora_config.r,
                    mlora_config.mlora_alpha,
                    mlora_config.mlora_dropout,
                    mlora_config.mlora_af,
                    mlora_config.mlora_af_sin_A,
                    mlora_config.mlora_af_sin_omega,
                    mlora_config.mlora_use_P,
                    mlora_config.mlora_use_b,
                    mlora_config.init_mlora_weights,
                )

            elif isinstance(target, MLoraLayer):
                target.update_layer(
                    adapter_name,
                    mlora_config.r,
                    mlora_config.mlora_alpha,
                    mlora_config.mlora_dropout,
                    mlora_config.mlora_af,
                    mlora_config.mlora_af_sin_A,
                    mlora_config.mlora_af_sin_omega,
                    mlora_config.mlora_use_P,
                    mlora_config.mlora_use_b,
                    mlora_config.init_mlora_weights,
                )
            else:
                new_module = self._create_new_module(mlora_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {mlora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "mlora_" in name:
                module.to(old_module.weight.device)
            if "ranknum" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, MLoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, MLoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        """
        This method merges the MLoRa layers into the base model.
        """
        for module in self.model.modules():
            if isinstance(module, MLoraLayer):
                module.merge()

    def unmerge_adapter(self):
        """
        This method unmerges the MLoRa layers from the base model.
        """
        for module in self.model.modules():
            if isinstance(module, MLoraLayer):
                module.unmerge()

    @staticmethod
    def _prepare_mlora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_MLORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_MLORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config

    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge MLORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "mlora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, MLoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(self, adapters, weights, adapter_name, combination_type="svd"):
        """
        This method adds a new adapter by merging the given adapters with the given weights.

        Args:
            adapters (list): List of adapter names to be merged.
            weights (list): List of weights for each adapter.
            adapter_name (str): Name of the new adapter.
            combination_type (str): Type of merging. Can be one of [`svd`, `linear`]
        """
        if adapter_name in list(self.peft_config.keys()):
            return
        for adapter in adapters:
            if adapter not in list(self.peft_config.keys()):
                raise ValueError(f"Adapter {adapter} does not exist")

        # if there is only one adapter, we can only use linear merging
        combination_type = "linear" if len(adapters) == 1 else combination_type

        # new rank is the max of all ranks of the adapters
        r_0 = self.peft_config[adapters[0]].r
        layer_num = len(r_0)
        new_rank = r_0
        for adapter in adapters:
            if combination_type == "linear":
                if self.peft_config[adapter].r != r_0:
                    raise ValueError("All adapters must have the same r value when using `linear` combination_type")
            elif combination_type == "svd":
                if len(self.peft_config[adapter].r) != layer_num:
                    raise ValueError("All adapters must have the same dims of r")
                for i in range(layer_num):
                    new_rank[i] = max(r_0[i], self.peft_config[adapter].r[i])
            else:
                raise ValueError(f"Invalid combination_type: {combination_type}")

        self.peft_config[adapter_name] = replace(self.peft_config[adapters[0]], r=new_rank, mlora_alpha=new_rank[-1])
        self._find_and_replace(adapter_name)
        mark_only_mlora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        _freeze_adapter(self.model, adapter_name)
        key_list = [key for key, _ in self.model.named_modules() if "mlora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, MLoraLayer):
                if layer_num > 1:
                    if adapter_name in target.mlora_A:
                        target_mlora_A = target.mlora_A[adapter_name]
                        target_mlora_B = target.mlora_B[adapter_name]
                    elif adapter_name in target.mlora_embedding_A:
                        target_mlora_A = target.mlora_embedding_A[adapter_name]
                        target_mlora_B = target.mlora_embedding_B[adapter_name]
                    target_mlora_A.data = target_mlora_A.data * 0.0
                    target_mlora_B.data = target_mlora_B.data * 0.0
                    if adapter_name in target.mlora_A:
                        variables = target.__dict__['_modules']
                        if target.mlora_use_P:
                            for i in range(1, layer_num):
                                variables["mlora_P_" + str(i)][adapter_name].data = (
                                        variables["mlora_P_" + str(i)][adapter_name].data * 0.0
                                )
                        if target.mlora_use_b:
                            for i in range(1, layer_num):
                                variables["mlora_b_" + str(i)][adapter_name].data = (
                                        variables["mlora_b_" + str(i)][adapter_name].data * 0.0
                                )
                    elif adapter_name in target.mlora_embedding_A:
                        variables = target.__dict__['_modules']
                        if target.mlora_use_P:
                            for i in range(1, layer_num):
                                variables["mlora_embedding_P_" + str(i)][adapter_name].data = (
                                        variables["mlora_embedding_P_" + str(i)][adapter_name].data * 0.0
                                )
                        if target.mlora_use_b:
                            for i in range(1, layer_num):
                                variables["mlora_embedding_b_" + str(i)][adapter_name].data = (
                                        variables["mlora_embedding_b_" + str(i)][adapter_name].data * 0.0
                                )

                if combination_type == "linear":
                    for adapter, weight in zip(adapters, weights):
                        if layer_num > 1:
                            if adapter in target.mlora_A:
                                current_adapter_mlora_A = target.mlora_A[adapter]
                                current_adapter_mlora_B = target.mlora_B[adapter]
                            elif adapter in target.mlora_embedding_A:
                                current_adapter_mlora_A = target.mlora_embedding_A[adapter]
                                current_adapter_mlora_B = target.mlora_embedding_B[adapter]
                            target_mlora_A.data = target_mlora_A.data + current_adapter_mlora_A.data * weight
                            target_mlora_B.data = target_mlora_B.data + current_adapter_mlora_B.data
                            if adapter in target.mlora_A:
                                variables = target.__dict__['_modules']
                                if target.mlora_use_P:
                                    for i in range(1, layer_num):
                                        if i != layer_num - 1:
                                            variables["mlora_P_" + str(i)][adapter_name].data = (
                                                    variables["mlora_P_" + str(i)][adapter_name].data +
                                                    variables["mlora_P_" + str(i)][adapter].data * weight
                                            )
                                        else:
                                            variables["mlora_P_" + str(i)][adapter_name].data = (
                                                    variables["mlora_P_" + str(i)][adapter_name].data +
                                                    variables["mlora_P_" + str(i)][adapter].data * weight *
                                                    target.scaling[adapter]
                                            )
                                if target.mlora_use_b:
                                    for i in range(1, layer_num):
                                        if i != layer_num - 1:
                                            variables["mlora_b_" + str(i)][adapter_name].data = (
                                                    variables["mlora_b_" + str(i)][adapter_name].data +
                                                    variables["mlora_b_" + str(i)][adapter].data * weight
                                            )
                                        else:
                                            variables["mlora_b_" + str(i)][adapter_name].data = (
                                                    variables["mlora_b_" + str(i)][adapter_name].data +
                                                    variables["mlora_b_" + str(i)][adapter].data * weight *
                                                    target.scaling[adapter]
                                            )
                            elif adapter in target.mlora_embedding_A:
                                variables = target.__dict__['_modules']
                                if target.mlora_use_P:
                                    for i in range(1, layer_num):
                                        if i != layer_num - 1:
                                            variables["mlora_embedding_P_" + str(i)][adapter_name].data = (
                                                    variables["mlora_embedding_P_" + str(i)][adapter_name].data +
                                                    variables["mlora_embedding_P_" + str(i)][adapter].data * weight
                                            )
                                        else:
                                            variables["mlora_embedding_P_" + str(i)][adapter_name].data = (
                                                    variables["mlora_embedding_P_" + str(i)][adapter_name].data +
                                                    variables["mlora_embedding_P_" + str(i)][
                                                        adapter].data * weight * target.scaling[adapter]
                                            )
                                if target.mlora_use_b:
                                    for i in range(1, layer_num):
                                        if i != layer_num - 1:
                                            variables["mlora_embedding_b_" + str(i)][adapter_name].data = (
                                                    variables["mlora_embedding_b_" + str(i)][adapter_name].data +
                                                    variables["mlora_embedding_b_" + str(i)][adapter].data * weight
                                            )
                                        else:
                                            variables["mlora_embedding_b_" + str(i)][adapter_name].data = (
                                                    variables["mlora_embedding_b_" + str(i)][adapter_name].data +
                                                    variables["mlora_embedding_b_" + str(i)][
                                                        adapter].data * weight * target.scaling[adapter]
                                            )
                elif combination_type == "svd":
                    target_mlora_A.data, target_mlora_B.data = self._svd_weighted_adapter(
                        adapters, weights, new_rank, target, target_mlora_A, target_mlora_B
                    )

    def _svd_weighted_adapter(self, adapters, weights, new_rank, target, target_mlora_A, target_mlora_B):
        raise ValueError("[WARNING] This function has not been adjusted to MLoRA! This will be solved in the future version")
        delta_weight = weights[0] * target.get_delta_weight(adapters[0])
        for adapter, weight in zip(adapters[1:], weights[1:]):
            delta_weight += weight * target.get_delta_weight(adapter)
        conv2d = isinstance(target, Conv2d)
        if conv2d:
            conv2d_1x1 = target.weight.size()[2:4] == (1, 1)
            if not conv2d_1x1:
                delta_weight = delta_weight.flatten(start_dim=1)
            else:
                delta_weight = delta_weight.squeeze()
        if target.fan_in_fan_out:
            delta_weight = delta_weight.T

        U, S, Vh = torch.linalg.svd(delta_weight)
        U = U[:, :new_rank]
        S = S[:new_rank]
        U = U @ torch.diag(S)
        Vh = Vh[:new_rank, :]
        dist = torch.cat([U.flatten(), Vh.flatten()])
        hi_val = torch.quantile(dist, CLAMP_QUANTILE)
        low_val = -hi_val
        U = U.clamp(low_val, hi_val)
        Vh = Vh.clamp(low_val, hi_val)
        if conv2d:
            U = U.reshape(target_mlora_B.data.shape)
            Vh = Vh.reshape(target_mlora_A.data.shape)
        return Vh, U

    def delete_adapter(self, adapter_name):
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]
        key_list = [key for key, _ in self.model.named_modules() if "mlora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, MLoraLayer):
                mlora_layer_num = len(target.r)
                attr_list = [
                    "r",
                    "mlora_alpha",
                    "scaling",
                    "mlora_A",
                    "mlora_B",
                    "mlora_embedding_A",
                    "mlora_embedding_B",
                    "mlora_dropout",
                ]
                for i in range(1, mlora_layer_num):
                    attr_list.append("mlora_P_" + str(i))
                    attr_list.append("mlora_b_" + str(i))
                    attr_list.append("mlora_embedding_P_" + str(i))
                    attr_list.append("mlora_embedding_b_" + str(i))

                for attr in attr_list:
                    if adapter_name in getattr(target, attr):
                        getattr(target, attr).pop(adapter_name)
                if target.active_adapter == adapter_name:
                    resetting_active_adapter = list(self.peft_config.keys())[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to {resetting_active_adapter}. "
                    )
                    target.active_adapter = resetting_active_adapter

    def merge_and_unload(self):
        r"""
        This method merges the MLoRA layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge()

    def unload(self):
        """
        Gets back the base model by removing all the mlora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)


def mark_only_mlora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, MLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class MLoraLayer:
    def __init__(self, in_features: int, out_features: int, mlora_layer_num: int, **kwargs):
        self.r = {}
        '''
        if len(self.r) < 2:
            raise ValueError(
                "At least 2 rank values are necessary in MLoraModel."
            )
        '''
        self.mlora_alpha = {}
        self.scaling = {}
        self.mlora_dropout = nn.ModuleDict({})
        self.mlora_af = ""
        self.mlora_af_sin_A = 0.0005
        self.mlora_af_sin_omega = 1000
        self.mlora_use_P = True
        self.mlora_use_b = True
        variables = self.__dict__['_modules']
        for i in range(1, mlora_layer_num):
            variables["mlora_P_" + str(i)] = nn.ParameterDict({})
            variables["mlora_b_" + str(i)] = nn.ParameterDict({})
        self.mlora_A = nn.ParameterDict({})
        self.mlora_B = nn.ParameterDict({})
        # For Embedding layer
        # variables = self.__dict__['_modules']
        for i in range(1, mlora_layer_num):
            variables["mlora_embedding_P_" + str(i)] = nn.ParameterDict({})
            variables["mlora_embedding_b_" + str(i)] = nn.ParameterDict({})
        self.mlora_embedding_A = nn.ParameterDict({})
        self.mlora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(self, adapter_name, r, mlora_alpha, mlora_dropout, mlora_af, mlora_af_sin_A, mlora_af_sin_omega, mlora_use_P, mlora_use_b, init_mlora_weights):
        self.r[adapter_name] = r
        self.mlora_alpha[adapter_name] = mlora_alpha
        if mlora_dropout > 0.0:
            mlora_dropout_layer = nn.Dropout(p=mlora_dropout)
        else:
            mlora_dropout_layer = nn.Identity()

        self.mlora_dropout.update(nn.ModuleDict({adapter_name: mlora_dropout_layer}))

        self.mlora_af = mlora_af
        if mlora_af not in ACTIVATION_FUNCTION_MAPPING:
            raise ValueError(
                "The specific activation function is not supported in MLoRA."
            )
        self.mlora_af_sin_A = mlora_af_sin_A
        self.mlora_af_sin_omega = mlora_af_sin_omega
        self.mlora_use_P = mlora_use_P
        self.mlora_use_b = mlora_use_b
        # Actual trainable parameters
        layer_num = len(r)
        if layer_num > 1:
            if r[0] > 0 and r[1] > 0:
                self.mlora_A.update(
                    nn.ParameterDict({adapter_name: nn.Parameter(
                        nn.init.kaiming_uniform_(torch.empty(r[0], r[1]), a=math.sqrt(5)))})
                )
                self.mlora_B.update(
                    nn.ParameterDict({adapter_name: nn.Parameter(
                        nn.init.zeros_(torch.empty(self.out_features, r[0])))})
                )
        else:
            raise ValueError(
                "At least 2 rank values are necessary in MLoraModel."
            )
        variables = self.__dict__['_modules']
        for i in range(1, layer_num-1):
            if r[i] > 0 and r[i+1] >0:
                if mlora_use_P == True:
                    variables["mlora_P_" + str(i)].update(
                        nn.ParameterDict({adapter_name: nn.Parameter(
                            nn.init.kaiming_uniform_(torch.empty(r[i], r[i + 1]), a=math.sqrt(5)))})
                    )
                else:
                    if r[i] != self.out_features:
                        raise ValueError(
                            "If linear transformation (Ps) is not applied in MLoraModel, "
                            "all the values in r except r[0] must be identical with the output dimension value."
                        )
                if mlora_use_b:
                    variables["mlora_b_" + str(i)].update(
                        nn.ParameterDict({adapter_name: nn.Parameter(
                            nn.init.zeros_(torch.empty(r[i + 1], 1)))})
                    )
        if r[-1] > 0:
            if mlora_use_P == True:
                variables["mlora_P_" + str(layer_num - 1)].update(
                    nn.ParameterDict({adapter_name: nn.Parameter(
                        nn.init.zeros_(torch.empty(r[-1], self.in_features)))})
                )
            else:
                if r[-1] != self.out_features:
                    raise ValueError(
                        "If linear transformation (Ps) is not applied in MLoraModel, "
                        "all the values in r except r[0] must be identical with the output dimension value."
                    )
            if mlora_use_b:
                variables["mlora_b_" + str(layer_num - 1)].update(
                    nn.ParameterDict({adapter_name: nn.Parameter(
                        nn.init.zeros_(torch.empty(self.in_features, 1)))})
                )
            self.scaling[adapter_name] = mlora_alpha / r[0]
        if init_mlora_weights:
            self.reset_mlora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_conv2d(self, adapter_name, r, mlora_alpha, mlora_dropout, init_mlora_weights):
        raise ValueError("[WARNING] This function has not been adjusted to MLoRA! This will be solved in the future version")
        self.r[adapter_name] = r
        self.mlora_alpha[adapter_name] = mlora_alpha
        if mlora_dropout > 0.0:
            mlora_dropout_layer = nn.Dropout(p=mlora_dropout)
        else:
            mlora_dropout_layer = nn.Identity()

        self.mlora_dropout.update(nn.ModuleDict({adapter_name: mlora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            kernel_size = self.kwargs["kernel_size"]
            stride = self.kwargs["stride"]
            padding = self.kwargs["padding"]
            self.mlora_A.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)})
            )
            self.mlora_B.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)})
            )
            self.scaling[adapter_name] = mlora_alpha / r
        if init_mlora_weights:
            self.reset_mlora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_embedding(self, adapter_name, r, mlora_alpha, mlora_dropout, mlora_af, mlora_af_sin_A,
                               mlora_af_sin_omega, mlora_use_P, mlora_use_b, init_mlora_weights):
        self.r[adapter_name] = r
        self.mlora_alpha[adapter_name] = mlora_alpha
        if mlora_dropout > 0.0:
            mlora_dropout_layer = nn.Dropout(p=mlora_dropout)
        else:
            mlora_dropout_layer = nn.Identity()

        self.mlora_dropout.update(nn.ModuleDict({adapter_name: mlora_dropout_layer}))

        self.mlora_af = mlora_af
        if mlora_af not in ACTIVATION_FUNCTION_MAPPING:
            raise ValueError(
                "The specific activation function is not supported in MLoRA."
            )
        self.mlora_af_sin_A = mlora_af_sin_A
        self.mlora_af_sin_omega = mlora_af_sin_omega
        self.mlora_use_P = mlora_use_P
        self.mlora_use_b = mlora_use_b
        # Actual trainable parameters
        layer_num = len(r)
        if layer_num > 1:
            if r[0] > 0 and r[1] > 0:
                weight_A = torch.randn((r[0], r[1]), dtype=self.weight.dtype, device=self.weight.device)
                weight_B = torch.randn((self.out_features, r[0]), dtype=self.weight.dtype, device=self.weight.device)
                self.mlora_embedding_A.update(
                    nn.ParameterDict({adapter_name: nn.Parameter(weight_A)})
                )
                self.mlora_embedding_B.update(
                    nn.ParameterDict({adapter_name: nn.Parameter(weight_B)})
                )
        else:
            raise ValueError(
                "At least 2 rank values are necessary in MLoraModel."
            )
        variables = self.__dict__['_modules']
        for i in range(1, layer_num - 1):
            if r[i] > 0 and r[i + 1] > 0:
                if mlora_use_P == True:
                    weight_P = torch.randn((r[i], r[i + 1]), dtype=self.weight.dtype, device=self.weight.device)
                    variables["mlora_embedding_P_" + str(i)].update(
                        nn.ParameterDict({adapter_name: nn.Parameter(weight_P)})
                    )
                else:
                    if r[i] != self.out_features:
                        raise ValueError(
                            "If linear transformation (Ps) is not applied in MLoraModel, "
                            "all the values in r except r[0] must be identical with the output dimension value."
                        )
                if mlora_use_b:
                    weight_b = torch.randn((r[i + 1], 1), dtype=self.weight.dtype, device=self.weight.device)
                    variables["mlora_embedding_b_" + str(i)].update(
                        nn.ParameterDict({adapter_name: nn.Parameter(weight_b)})
                    )
        if r[-1] > 0:
            if mlora_use_P == True:
                weight_P = torch.randn((r[-1], self.in_features), dtype=self.weight.dtype, device=self.weight.device)
                variables["mlora_embedding_P_" + str(layer_num - 1)].update(
                    nn.ParameterDict({adapter_name: nn.Parameter(weight_P)})
                )
            else:
                if r[-1] != self.out_features:
                    raise ValueError(
                        "If linear transformation (Ps) is not applied in MLoraModel, "
                        "all the values in r except r[0] must be identical with the output dimension value."
                    )
            if mlora_use_b:
                weight_b = torch.randn((self.in_features, 1), dtype=self.weight.dtype, device=self.weight.device)
                variables["mlora_embedding_b_" + str(layer_num - 1)].update(
                    nn.ParameterDict({adapter_name: nn.Parameter(weight_b)})
                )
            self.scaling[adapter_name] = mlora_alpha / r[0]
        if init_mlora_weights:
            self.reset_mlora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_mlora_parameters(self, adapter_name):
        layer_num = len(self.r[adapter_name])
        if adapter_name in self.mlora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            if layer_num > 1:
                nn.init.kaiming_uniform_(self.mlora_A[adapter_name], a=math.sqrt(5))
                nn.init.zeros_(self.mlora_B[adapter_name])
            else:
                raise ValueError(
                    "At least 2 rank values are necessary in MLoraModel."
                )
            variables = self.__dict__['_modules']
            if self.mlora_use_P:
                for i in range(1, layer_num - 1):
                    nn.init.kaiming_uniform_(variables["mlora_P_" + str(i)][adapter_name], a=math.sqrt(5))
                nn.init.zeros_(variables["mlora_P_" + str(layer_num - 1)][adapter_name])
            if self.mlora_use_b:
                for i in range(1, layer_num - 1):
                    nn.init.zeros_(variables["mlora_b_" + str(i)][adapter_name])
                nn.init.zeros_(variables["mlora_b_" + str(layer_num - 1)][adapter_name])

        if adapter_name in self.mlora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            if layer_num > 1:
                nn.init.zeros_(self.mlora_embedding_A[adapter_name])
                nn.init.zeros_(self.mlora_embedding_B[adapter_name])
            else:
                raise ValueError(
                    "At least 2 rank values are necessary in MLoraModel."
                )
            variables = self.__dict__['_modules']
            if self.mlora_use_P:
                for i in range(1, layer_num):
                    nn.init.zeros_(variables["mlora_embedding_P_" + str(i)][adapter_name])
            if self.mlora_use_b:
                for i in range(1, layer_num):
                    nn.init.zeros_(variables["mlora_embedding_b_" + str(i)][adapter_name])


class Linear(nn.Linear, MLoraLayer):
    # MLora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: List = [],
        mlora_alpha: int = 1,
        mlora_dropout: float = 0.0,
        mlora_af: str = "Sigmoid",
        mlora_af_sin_A: float = 0.0005,
        mlora_af_sin_omega: float = 1000,
        mlora_use_P: bool = True,
        mlora_use_b: bool = True,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        init_mlora_weights = kwargs.pop("init_mlora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        MLoraLayer.__init__(self, in_features=in_features, out_features=out_features, mlora_layer_num=len(r))
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, mlora_alpha, mlora_dropout, mlora_af, mlora_af_sin_A, mlora_af_sin_omega, mlora_use_P, mlora_use_b, init_mlora_weights)
        self.active_adapter = adapter_name
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self):
        if self.active_adapter not in self.mlora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        layer_num = len(self.r[self.active_adapter])
        if layer_num > 1:
            self.weight.data = self.weight.data + self.get_delta_weight(self.active_adapter)
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.mlora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        layer_num = len(self.r[self.active_adapter])
        if layer_num > 1:
            self.weight.data -= self.get_delta_weight(self.active_adapter)
            self.merged = False

    def get_delta_weight(self, adapter):
        layer_num = len(self.r[adapter])
        mlora_output = transpose(self.mlora_B[adapter] @ self.mlora_A[adapter], True)
        variables = self.__dict__['_modules']
        for i in range(1, layer_num):
            if self.mlora_af == "Sin":
                af = ACTIVATION_FUNCTION_MAPPING[self.mlora_af](self.mlora_af_sin_A, self.mlora_af_sin_omega)
            else:
                af = ACTIVATION_FUNCTION_MAPPING[self.mlora_af]()
            mlora_output = af(mlora_output)
            if self.mlora_use_P:
                mlora_output = transpose(variables["mlora_P_" + str(i)][adapter], True) @ mlora_output
            if self.mlora_use_b:
                mlora_output = mlora_output + variables["mlora_b_" + str(i)][adapter]
        return mlora_output * self.scaling[adapter]

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.mlora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        layer_num = len(self.r[self.active_adapter])
        if self.disable_adapters:
            if layer_num > 1 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif layer_num > 1 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.mlora_A[self.active_adapter].dtype)

            mlora_output = transpose(self.mlora_B[self.active_adapter] @ self.mlora_A[self.active_adapter], True)
            variables = self.__dict__['_modules']
            for i in range(1, layer_num):
                if self.mlora_af == "Sin":
                    af = ACTIVATION_FUNCTION_MAPPING[self.mlora_af](self.mlora_af_sin_A, self.mlora_af_sin_omega)
                else:
                    af = ACTIVATION_FUNCTION_MAPPING[self.mlora_af]()
                mlora_output = af(mlora_output)
                if self.mlora_use_P:
                    mlora_output = transpose(variables["mlora_P_" + str(i)][self.active_adapter], True) @ mlora_output
                if self.mlora_use_b:
                    mlora_output = mlora_output + variables["mlora_b_" + str(i)][self.active_adapter]
            mlora_output = self.mlora_dropout[self.active_adapter](x) @ mlora_output
            result = result + mlora_output * self.scaling[self.active_adapter]
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result


class Embedding(nn.Embedding, MLoraLayer):
    # MLoRA implemented in a Embedding layer
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        r: List = [],
        mlora_alpha: int = 1,
        mlora_dropout: float = 0.0,
        mlora_af: str = "Sigmoid",
        mlora_af_sin_A: float = 0.0005,
        mlora_af_sin_omega: float = 1000,
        mlora_use_P: bool = True,
        mlora_use_b: bool = True,
        **kwargs,
    ):
        init_mlora_weights = kwargs.pop("init_mlora_weights", True)

        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        MLoraLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim, mlora_layer_num=len(r))

        self.weight.requires_grad = False

        nn.Embedding.reset_parameters(self)
        self.update_layer_embedding(adapter_name, r, mlora_alpha, mlora_dropout, mlora_af, mlora_af_sin_A, mlora_af_sin_omega, mlora_use_P, mlora_use_b, init_mlora_weights)
        self.active_adapter = adapter_name

    def unmerge(self, mode: bool = True):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        layer_num = len(self.r[self.active_adapter])
        if layer_num > 1:
            self.weight.data -= self.get_delta_weight(self.active_adapter)
            self.merged = False

    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        layer_num = len(self.r[self.active_adapter])
        if layer_num > 1:
            self.weight.data = self.weight.data + self.get_delta_weight(self.active_adapter)
            self.merged = True

    def get_delta_weight(self, adapter):
        layer_num = len(self.r[adapter])
        mlora_output = transpose(
            self.mlora_embedding_B[adapter] @ self.mlora_embedding_A[adapter], True)
        variables = self.__dict__['_modules']
        for i in range(1, layer_num):
            if self.mlora_af == "Sin":
                af = ACTIVATION_FUNCTION_MAPPING[self.mlora_af](self.mlora_af_sin_A, self.mlora_af_sin_omega)
            else:
                af = ACTIVATION_FUNCTION_MAPPING[self.mlora_af]()
            mlora_output = af(mlora_output)
            if self.mlora_use_P:
                mlora_output = transpose(variables["mlora_embedding_P_" + str(i)][adapter],
                                         True) @ mlora_output
            if self.mlora_use_b:
                mlora_output = mlora_output + variables["mlora_embedding_b_" + str(i)][adapter]
        return mlora_output * self.scaling[adapter]

    def forward(self, x: torch.Tensor):
        layer_num = len(self.r[self.active_adapter])
        variables = self.__dict__['_modules']
        if self.disable_adapters:
            if layer_num > 1 and self.merged:
                self.unmerge()
            return nn.Embedding.forward(self, x)

        elif layer_num > 1 and not self.merged:
            result = nn.Embedding.forward(self, x)
            mlora_output = transpose(
                self.mlora_embedding_B[self.active_adapter] @ self.mlora_embedding_A[self.active_adapter], True)
            for i in range(1, layer_num):
                if self.mlora_af == "Sin":
                    af = ACTIVATION_FUNCTION_MAPPING[self.mlora_af](self.mlora_af_sin_A, self.mlora_af_sin_omega)
                else:
                    af = ACTIVATION_FUNCTION_MAPPING[self.mlora_af]()
                mlora_output = af(mlora_output)
                if self.mlora_use_P:
                    mlora_output = transpose(variables["mlora_embedding_P_" + str(i)][self.active_adapter],
                                             True) @ mlora_output
                if self.mlora_use_b:
                    mlora_output = mlora_output + variables["mlora_embedding_b_" + str(i)][self.active_adapter]
            after_mlora = F.embedding(
                x,
                mlora_output,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            result = result + after_mlora * self.scaling[self.active_adapter]
            return result
        else:
            return nn.Embedding.forward(self, x)


class Conv2d(nn.Conv2d, MLoraLayer):
    # MLora implemented in a conv2d layer
    def __init__(
        self,
        adapter_name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        r: int = 0,
        mlora_alpha: int = 1,
        mlora_dropout: float = 0.0,
        **kwargs,
    ):
        raise ValueError("[WARNING] This function has not been adjusted to MLoRA! This will be solved in the future version")
        init_mlora_weights = kwargs.pop("init_mlora_weights", True)

        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding)
        MLoraLayer.__init__(
            self,
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        nn.Conv2d.reset_parameters(self)
        self.update_layer_conv2d(adapter_name, r, mlora_alpha, mlora_dropout, init_mlora_weights)
        self.active_adapter = adapter_name

    def merge(self):
        raise ValueError("[WARNING] This function has not been adjusted to MLoRA! This will be solved in the future version")
        if self.active_adapter not in self.mlora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += self.get_delta_weight(self.active_adapter)
            self.merged = True

    def unmerge(self):
        raise ValueError("[WARNING] This function has not been adjusted to MLoRA! This will be solved in the future version")
        if self.active_adapter not in self.mlora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= self.get_delta_weight(self.active_adapter)
            self.merged = False

    def get_delta_weight(self, adapter):
        raise ValueError("[WARNING] This function has not been adjusted to MLoRA! This will be solved in the future version")
        if self.weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            return (
                self.mlora_B[adapter].weight.squeeze(3).squeeze(2) @ self.mlora_A[adapter].weight.squeeze(3).squeeze(2)
            ).unsqueeze(2).unsqueeze(3) * self.scaling[adapter]
        else:
            # conv2d 3x3
            return (
                F.conv2d(
                    self.mlora_A[adapter].weight.permute(1, 0, 2, 3),
                    self.mlora_B[adapter].weight,
                ).permute(1, 0, 2, 3)
                * self.scaling[adapter]
            )

    def forward(self, x: torch.Tensor):
        raise ValueError("[WARNING] This function has not been adjusted to MLoRA! This will be solved in the future version")
        previous_dtype = x.dtype

        if self.active_adapter not in self.mlora_A.keys():
            return F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            x = x.to(self.mlora_A[self.active_adapter].weight.dtype)

            result += (
                self.mlora_B[self.active_adapter](
                    self.mlora_A[self.active_adapter](self.mlora_dropout[self.active_adapter](x))
                )
                * self.scaling[self.active_adapter]
            )
        else:
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        result = result.to(previous_dtype)

        return result


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, MLoraLayer):
        # MLora implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: List = [],
            mlora_alpha: int = 1,
            mlora_dropout: float = 0.0,
            mlora_af: str = "Sigmoid",
            mlora_af_sin_A: float = 0.0005,
            mlora_af_sin_omega: float = 1000,
            mlora_use_P: bool = True,
            mlora_use_b: bool = True,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            MLoraLayer.__init__(self, in_features=in_features, out_features=out_features, mlora_layer_num=len(r))

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_mlora_weights = kwargs.pop("init_mlora_weights", True)
            self.update_layer(adapter_name, r, mlora_alpha, mlora_dropout, mlora_af, mlora_af_sin_A, mlora_af_sin_omega, mlora_use_P, mlora_use_b, init_mlora_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            layer_num = len(self.r[self.active_adapter])
            if self.disable_adapters or self.active_adapter not in self.mlora_A.keys():
                return result
            elif layer_num > 1:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()

                    mlora_output = transpose(self.mlora_B[self.active_adapter] @ self.mlora_A[self.active_adapter], True)
                    variables = self.__dict__['_modules']
                    for i in range(1, layer_num):
                        if self.mlora_af == "Sin":
                            af = ACTIVATION_FUNCTION_MAPPING[self.mlora_af](self.mlora_af_sin_A,
                                                                            self.mlora_af_sin_omega)
                        else:
                            af = ACTIVATION_FUNCTION_MAPPING[self.mlora_af]()
                        mlora_output = af(mlora_output)
                        if self.mlora_use_P:
                            mlora_output = transpose(variables["mlora_P_" + str(i)][self.active_adapter], True) @ mlora_output
                        if self.mlora_use_b:
                            mlora_output = mlora_output + variables["mlora_b_" + str(i)][self.active_adapter]
                    mlora_output = self.mlora_dropout[self.active_adapter](x) @ mlora_output
                    mlora_output = mlora_output.to(expected_dtype)
                    mlora_output = mlora_output * self.scaling[self.active_adapter]
                else:
                    mlora_output = transpose(self.mlora_B[self.active_adapter] @ self.mlora_A[self.active_adapter], True)
                    variables = self.__dict__['_modules']
                    for i in range(1, layer_num):
                        if self.mlora_af == "Sin":
                            af = ACTIVATION_FUNCTION_MAPPING[self.mlora_af](self.mlora_af_sin_A,
                                                                            self.mlora_af_sin_omega)
                        else:
                            af = ACTIVATION_FUNCTION_MAPPING[self.mlora_af]()
                        mlora_output = af(mlora_output)
                        if self.mlora_use_P:
                            mlora_output = transpose(variables["mlora_P_" + str(i)][self.active_adapter], True) @ mlora_output
                        if self.mlora_use_b:
                            mlora_output = mlora_output + variables["mlora_b_" + str(i)][self.active_adapter]
                    mlora_output = self.mlora_dropout[self.active_adapter](x) @ mlora_output
                    mlora_output = mlora_output * self.scaling[self.active_adapter]
                result = result + mlora_output
            return result

    if is_bnb_4bit_available():

        class Linear4bit(bnb.nn.Linear4bit, MLoraLayer):
            # MLora implemented in a dense layer
            def __init__(
                    self,
                    adapter_name,
                    in_features,
                    out_features,
                    r: List = [],
                    mlora_alpha: int = 1,
                    mlora_dropout: float = 0.0,
                    mlora_af: str = "Sigmoid",
                    mlora_af_sin_A: float = 0.0005,
                    mlora_af_sin_omega: float = 1000,
                    mlora_use_P: bool = True,
                    mlora_use_b: bool = True,
                    **kwargs,
            ):
                bnb.nn.Linear4bit.__init__(
                    self,
                    in_features,
                    out_features,
                    bias=kwargs.get("bias", True),
                    compute_dtype=kwargs.get("compute_dtype", torch.float32),
                    compress_statistics=kwargs.get("compress_statistics", True),
                    quant_type=kwargs.get("quant_type", "nf4"),
                )
                MLoraLayer.__init__(self, in_features=in_features, out_features=out_features, mlora_layer_num=len(r))

                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False

                init_mlora_weights = kwargs.pop("init_mlora_weights", True)
                self.update_layer(adapter_name, r, mlora_alpha, mlora_dropout, mlora_af, mlora_af_sin_A,
                                  mlora_af_sin_omega, mlora_use_P, mlora_use_b, init_mlora_weights)
                self.active_adapter = adapter_name

            def forward(self, x: torch.Tensor):
                result = super().forward(x)

                layer_num = len(self.r[self.active_adapter])
                if self.disable_adapters or self.active_adapter not in self.mlora_A.keys():
                    return result
                elif layer_num > 1:
                    result = result.clone()
                    if not torch.is_autocast_enabled():
                        expected_dtype = result.dtype
                        x = x.to(self.mlora_A[self.active_adapter].dtype)
                        mlora_output = transpose(self.mlora_B[self.active_adapter] @ self.mlora_A[self.active_adapter],
                                                 True)
                        variables = self.__dict__['_modules']
                        for i in range(1, layer_num):
                            if self.mlora_af == "Sin":
                                af = ACTIVATION_FUNCTION_MAPPING[self.mlora_af](self.mlora_af_sin_A,
                                                                                self.mlora_af_sin_omega)
                            else:
                                af = ACTIVATION_FUNCTION_MAPPING[self.mlora_af]()
                            mlora_output = af(mlora_output)
                            if self.mlora_use_P:
                                mlora_output = transpose(variables["mlora_P_" + str(i)][self.active_adapter],
                                                         True) @ mlora_output
                            if self.mlora_use_b:
                                mlora_output = mlora_output + variables["mlora_b_" + str(i)][self.active_adapter]
                        mlora_output = self.mlora_dropout[self.active_adapter](x) @ mlora_output
                        mlora_output = mlora_output.to(expected_dtype)
                        mlora_output = mlora_output * self.scaling[self.active_adapter]
                    else:
                        mlora_output = transpose(self.mlora_B[self.active_adapter] @ self.mlora_A[self.active_adapter],
                                                 True)
                        variables = self.__dict__['_modules']
                        for i in range(1, layer_num):
                            if self.mlora_af == "Sin":
                                af = ACTIVATION_FUNCTION_MAPPING[self.mlora_af](self.mlora_af_sin_A,
                                                                                self.mlora_af_sin_omega)
                            else:
                                af = ACTIVATION_FUNCTION_MAPPING[self.mlora_af]()
                            mlora_output = af(mlora_output)
                            if self.mlora_use_P:
                                mlora_output = transpose(variables["mlora_P_" + str(i)][self.active_adapter],
                                                         True) @ mlora_output
                            if self.mlora_use_b:
                                mlora_output = mlora_output + variables["mlora_b_" + str(i)][self.active_adapter]
                        mlora_output = self.mlora_dropout[self.active_adapter](x) @ mlora_output
                        mlora_output = mlora_output * self.scaling[self.active_adapter]
                    result = result + mlora_output
                return result

