from transformers import LlamaConfig
from typing import List, Union


class CompressedLlamaConfig(LlamaConfig):

    def __init__(
        self,
        share_layers: Union[List[List[int]], str] = "none",
        **kwargs,
    ):
        if isinstance(share_layers, str) and share_layers not in ["none", "all"]:
            raise ValueError(f"`share_layers` must be 'none' or all', got {share_layers}.")
        if isinstance(share_layers, list):
            already_shared = []
            # check all elements are of type list
            for shared_layer in share_layers:
                if not isinstance(shared_layer, list):
                    raise ValueError(f"`share_layers` must be contain a list of list of ints, got {share_layers}.")
                for layer in shared_layer:
                    if not isinstance(layer, int):
                        raise ValueError(f"`share_layers` must be contain a list of list of ints, got {share_layers}.")
                    if layer in already_shared:
                        raise ValueError(f"you can only share a lyaer once, got {share_layers}.")
                    already_shared.append(layer)

        self.share_layers = share_layers
        super().__init__(**kwargs)
