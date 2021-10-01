"""
Astrocyte model functional portion
"""
from typing import NamedTuple
import torch


class AstroLearningParams(NamedTuple):
    upper_thr: torch.Tensor = torch.as_tensor(2)
    lower_thr: torch.tensor = torch.as_tensor(1)


class AstroActivityParams(NamedTuple):
    target: torch.Tensor = torch.as_tensor(0.1)
    alpha: torch.Tensor = torch.as_tensor(1.0)


class AstroParams(NamedTuple):
    tau: torch.Tensor = torch.as_tensor(1e-3)
    alpha: torch.Tensor = torch.as_tensor(1e-3)
    learning_params: AstroLearningParams = None
    activity_params: AstroActivityParams = None

    

class AstroState(NamedTuple):
    """
    s: state variable
    """

    pre: torch.Tensor = torch.as_tensor(0.0)
    post: torch.Tensor = torch.as_tensor(0.0)


def _set_module_weights(module, bounding_func=None, new_weight=None, add=None, mult=None):
    if new_weight:
        w = new_weight
            
    elif add:
        w = module.weights.detach()
        w += add
            
    elif mult:
        w = module.weights.detach()
        w = w * mult

    if bounding_func: w = bounding_func(w)
    module.weights.data = w
    
    
def astro_step(pre, post, params, state):
    if state is None:
        state = AstroState(pre=torch.zeros_like(pre), post=torch.zeros_like(post))
        
    state.pre += torch.mean(pre) * params.alpha - state.pre * params.tau * dt
    state.post += torch.mean(post) * params.alpha - state.pre * params.tau * dt

    return state


def astrocyte_effect_learning(state):
    pass


def astro_get_presynaptic_current(state, params):
    cur = params.activity_params.alpha * (params.activity_params.target - state.post)

    return cur
