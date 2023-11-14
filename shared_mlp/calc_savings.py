import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
from torch import nn
from datasets import load_dataset, load_from_disk
from datasets.fingerprint import Hasher

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.testing_utils import CaptureLogger
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.34.0")

require_version("datasets>=2.14.0", "To fix: pip install -r requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization"
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (model_args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (model_args,) = parser.parse_args_into_dataclasses()

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name)

    print("calculating num of params in original model")
    og_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # grab an mlp
    main_mlp = model.get_decoder().layers[0].mlp

    # average weights and activations
    print("averaging weights and activations")
    for layer in model.get_decoder().layers[1:]:
        mm_sd = main_mlp.state_dict()
        lm_sd = layer.mlp.state_dict()
        new_sd = {key: (mm_sd[key] + lm_sd[key]) / 2.0 for key in mm_sd}
        main_mlp.load_state_dict(new_sd)
    print(main_mlp)

    # replace mlps in layers with shared mlp
    print("replacing mlps")
    for layer in model.get_decoder().layers:
        layer.mlp = main_mlp

    # calculate new param count
    print("calculating num of params in new model")
    new_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"og_params: {og_params_count}, new_params: {new_params_count}, {round(1 - (new_params_count / og_params_count), 3)}% smaller")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
