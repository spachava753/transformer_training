from transformers import LlamaConfig
from typing import List, Union, Dict


class CompressedLlamaConfig(LlamaConfig):

    def __init__(
        self,
        tied_layers: Union[Dict[int, int], str] = "none",
        **kwargs,
    ):
        if isinstance(tied_layers, str) and tied_layers not in ["none", "all"]:
            raise ValueError(f"`tied_layers` must be 'none' or all', got {tied_layers}.")
        if isinstance(tied_layers, dict):
            # check all elements are of type list
            for layer, tied_layer in tied_layers.items():
                if not isinstance(layer, int) or not isinstance(tied_layer, int):
                    raise ValueError(f"`tied_layers` must be a dict of ints to ints, got {tied_layers}.")
                if layer < tied_layer:
                    raise ValueError(f"`tied_layers` must be a dict of ints to ints, where the key is <= value got {tied_layers}.")

        self.tied_layers = tied_layers
        super().__init__(**kwargs)
