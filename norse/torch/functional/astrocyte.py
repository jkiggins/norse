"""
Astrocyte model functional portion
"""

import torch

class AstroParameters(NamedTuple):
    """
    tau_pre (torch.Tensor): factor for internal state update from pre-synaptic activity
    tau_post (torch.Tensor): factor for internal state update from post-synaptic activity
    state_leak (torch.Tensor): Rate at which state regresses to its default value
    """

    tau_pre: torch.Tensor = torch.as_tensor(1e-3)
    tau_post: torch.Tensor = torch.as_tensor(1e-3)
    state_leak: torch.Tensor = torch.as_tensor(1e-4)
    spike_aggregator: str = "average"


class AstroState(NamedTuple):
    """
    s: state variable
    """

    s: torch.Tensor = torch.as_tensor(0.0)


def astro_neuron_step(pre_input_tensor, post_input_tensor, params, state):
    if params['spike_aggregator'] == 'average':
        pre_val = pre_input_tensor.mean()
        post_val = post_input_tensor.mean()

    state 
    
