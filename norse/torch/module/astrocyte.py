"""
NN Module that models (to a degree) astrocyte behavior
"""

from norse.torch.functional.astrocyte import (
    astro_state_prop_inc_exp_decay,
    astro_state_exp_avg,
    astro_proportional_target_effect,
    astro_proportional_effect,
)

from norse.torch.utils import registry, config

class Astrocyte:
    def __init__(self, params, state_fn, effect_fn, dt=0.001):
        self.params = params
        self.dt = dt
        self.state_fn = state_fn
        self.effect_fn = effect_fn


    def from_cfg(cfg):
        effect_params = cfg['astro_params'].deref('effect_params')

        params = {
            'alpha': cfg['astro_params']['alpha'],
            'tau': cfg['astro_params']['tau'],
            'effect_params': effect_params.as_dict()
        }

        state_fn = registry.get_entry(cfg['astro_params']['state_update_algo'])
        effect_fn = registry.get_entry(params['effect_params']['effect_algo'])
        
        return Astrocyte(params, state_fn, effect_fn, dt=cfg['sim']['dt'])
        

    def _effect(self, state):
        return self.effect_fn(state, self.params)        

    def forward(self, z, state):
        state = self.state_fn(z, self.params, state, dt=self.dt)


        return self._effect(state)


registry.add_entry("astro_state_prop_inc_exp_decay", astro_state_prop_inc_exp_decay)
registry.add_entry("astro_state_exp_avg", astro_state_prop_inc_exp_decay)
registry.add_entry("astro_proportional_effect", astro_proportional_effect)
registry.add_entry("astro_proportional_target_effect", astro_proportional_target_effect)
