"""
Astrocyte model functional portion
"""
from typing import NamedTuple
import torch


# Params
# class AstroActivityParams(NamedTuple):
#     # Proportional effect
#     target: torch.Tensor = torch.as_tensor(0.1)
#     alpha: torch.Tensor = torch.as_tensor(1.0)

#     # Constant
#     thr: torch.Tensor = torch.as_tensor(5)
#     const_effect: torch.Tensor = torch.as_tensor(0.5)

#     # Increment
#     inc_low_thr: torch.Tensor = torch.as_tensor(2)
#     inc_step: torch.Tensor = torch.as_tensor(0.1)
#     max_e: torch.Tensor = torch.as_tensor(1.0)
    
#     dec_high_thr: torch.Tensor = torch.as_tensor(8)
#     dec_step: torch.Tensor = torch.as_tensor(0.3)
#     min_e: torch.Tensor = torch.as_tensor(0.0)


# class AstroParams(NamedTuple):
#     tau: torch.Tensor = torch.as_tensor(1e-3)
#     alpha: torch.Tensor = torch.as_tensor(1e-3)
#     alpha_long: torch.Tensor = torch.as_tensor(1e-3)
#     activity_params: AstroActivityParams = None

    
# class AstroState(NamedTuple):
#     t_z: torch.Tensor = torch.as_tensor(0.0)
#     t_z_long: torch.Tensor = torch.as_tensor(0.0)
#     effect: torch.Tensor = torch.as_tensor(0.0)
    
    
# State Update
def astro_state_prop_inc_exp_decay(z, params, state, dt=0.001):
    if state is None:
        print("Astro state is None")
        state = {'t_z': 0.0}

    if len(z) > 1:
        z = float(z.mean())

    t_z_new = state['t_z'] + z * params['alpha'] - state['t_z'] * params['tau'] * dt
    
    state['t_z'] = t_z_new

    return state


def astro_state_exp_avg(z, params, state, dt=0.001):
    if state is None:
        print("Astro state is None")
        state = {'t_z1': 0.0, 't_z2': 0.0}

    if len(z) > 1:
        z = float(z.mean())

    t_z1_new = state['t_z1'] * (1.0 - params['alpha1']) + z * params['alpha1']
    t_z2_new = state['t_z2'] * (1.0 - params['alpha2']) + z * params['alpha2']
    
    state['t_z1'] = t_z1_new
    state['t_z2'] = t_z2_new

    return state


# Effect
def astro_proportional_target_effect(state, params):
    cur = params['effect_params']['alpha'] * (params['effect_params']['target'] - state['t_z'])
    
    return cur, state


def astro_proportional_effect(state, params):
    s = state['t_z2'] - (torch.abs(state['t_z2'] - state['t_z1']))

    cur_supress = params['effect_params']['alpha'] * s

    return cur_supress, state


# def astro_const_effect(state, params):
#     # If the activity is above a threshold, apply a constant current
#     if state.t_z > params.activity_params.thr:
#         cur = params.activity_params.const_effect
#     else:
#         cur = torch.as_tensor(0.0)

#     return cur, state


# def astro_inc_dec_effect(state, params):
#     activity_params = params.activity_params

#     effect = state.effect

#     if state.t_z < activity_params.inc_low_thr:
#         effect = effect + activity_params.inc_step
#         effect = torch.as_tensor(min(activity_params.max_e, effect))

#     elif state.t_z > activity_params.dec_high_thr:
#         effect = effect - activity_params.dec_step
#         effect = torch.as_tensor(max(activity_params.min_e, effect))

#     new_state = AstroState(t_z=state.t_z, effect=effect)

#     return effect, new_state
