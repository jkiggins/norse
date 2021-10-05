"""
Astrocyte model functional portion
"""
from typing import NamedTuple
import torch


# Params
class AstroActivityParams(NamedTuple):
    target: torch.Tensor = torch.as_tensor(0.1)
    alpha: torch.Tensor = torch.as_tensor(1.0)
    thr: torch.Tensor = torch.as_tensor(5)
    const_effect: torch.Tensor = torch.as_tensor(0.5)


class AstroParams(NamedTuple):
    tau: torch.Tensor = torch.as_tensor(1e-3)
    alpha: torch.Tensor = torch.as_tensor(1e-3)
    activity_params: AstroActivityParams = None

    
class AstroState(NamedTuple):
    t_z: torch.Tensor = torch.as_tensor(0.0)
    
    
# State Update
def astro_state_prop_inc_exp_decay(z, params, state, dt=0.001):
    if state is None:
        state = AstroState(t_z=torch.zeros_like(z))

    t_z_new = state.t_z + z * params.alpha - state.t_z * params.tau * dt

    return AstroState(t_z = t_z_new)


# Effect
def astro_porportional_presynaptic_current(state, params):
    cur = params.activity_params.alpha * (params.activity_params.target - state.t_z)

    return cur


def astro_const_presynaptic_current(state, params):
    # If the activity is above a threshold, apply a constant current
    if state.t_z > params.activity_params.thr:
        cur = params.activity_params.const_effect
    else:
        cur = 0.0

    return cur
    
    
    
