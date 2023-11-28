import logging
from dataclasses import dataclass, field

from transformers import AutoTokenizer, AutoModelForCausalLM
from compressed_llama.configuration_compressed_llama import CompressedLlamaConfig
from compressed_llama.modeling_compressed_llama import CompressedLlamaModel, CompressedLlamaForCausalLM

check_min_version("4.34.0")

logger = logging.getLogger(__name__)

@dataclass
class ScriptArguments:
    """
    Arguments pertaining to which model we are going to compress.
    """

    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization"
            )
        },
    )
    repo_id: Optional[str] = field(
        default=None,
        metadata={"help": "your repo id"},
    ),
    commit_message: Optional[str] = field(
        default=None,
        metadata={"help": "the commit message when you push your model"},
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

def compress_all(compressed_model, model):
    single_shared_mlp = compressed_model.model.layers[0].mlp
    cloned_mlp_state_dict = {name: param.clone() for name, param in model.model.layers[0].mlp.state_dict().items()}
    single_shared_mlp.load_state_dict(cloned_mlp_state_dict)
    for layer in open_llama_model.model.layers[1:]:
        ss_sd = single_shared_mlp.state_dict()
        lm_sd = layer.mlp.state_dict()
        new_sd = {key: (ss_sd[key] + lm_sd[key]) / 2.0 for key in ss_sd}
        single_shared_mlp.load_state_dict(new_sd)
    single_shared_mlp.state_dict()

def main():
    parser = HfArgumentParser((ScriptArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (script_args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (script_args,) = parser.parse_args_into_dataclasses()

    # TODO: check that imported_model is LlamaForCausalLM
    imported_model = AutoModelForCausalLM.from_pretrained(script_args.model_name)
    # open_llama_model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2")

    # TODO: make sure import config from imported_model
    config = CompressedLlamaConfig(share_layers='none', 
                                   bos_token_id=1, 
                                   eos_token_id=2, 
                                   hidden_act="silu", 
                                   hidden_size=3200, 
                                   initializer_range=0.02, 
                                   intermediate_size=8640,
                                   max_position_embeddings=2048,
                                   model_type="compressed_llama",
                                   num_attention_heads=32,
                                   num_hidden_layers=26,
                                   pad_token_id=0,
                                   rms_norm_eps=1e-06,
                                   tie_word_embeddings=False,
                                   torch_dtype="float16",
                                   transformers_version="4.31.0.dev0",
                                   use_cache=True,
                                   vocab_size=32000)
    
    model = CompressedLlamaForCausalLM(config)

    # load all of the rest of the parameter weights
    model.load_state_dict(imported_model.state_dict())

    # TODO: add support for compressing specific layers
    if share_layers == 'all':
        compress_all(model)
    
    model.push_to_hub(script_args.repo_id, commit_message=script_args.commit_message)

if __name__ == "__main__":
    main()
