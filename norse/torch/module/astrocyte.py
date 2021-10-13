"""
NN Module that models (to a degree) astrocyte behavior
"""

from norse.torch.functional.astrocyte import (
    AstroParams,
    AstroActivityParams,
    astro_state_prop_inc_exp_decay,
    astro_proportional_effect,
    astro_const_effect,
    astro_inc_dec_effect,
)

from norse.torch.utils import registry, config

class Astrocyte:
    def __init__(self, params, state_fn, effect_fn, dt=0.001):
        self.params = params
        self.dt = dt
        self.state_fn = state_fn
        self.effect_fn = effect_fn


    def from_cfg(cfg):
        activity_cfg = cfg['astro_params'].deref('activity_params')
        
        effect_fn_name = activity_cfg['affect_algo']
        effect_fn = registry.get_entry(effect_fn_name)

        state_fn_name = cfg['astro_params']['state_update_algo']
        state_fn = registry.get_entry(state_fn_name)

        if effect_fn_name == "astro_const_effect":
            activity_params = AstroActivityParams(
                thr=activity_cfg['thr'],
                const_effect=activity_cfg['const_effect'])

        elif effect_fn_name == "astro_proportional_effect":
            activity_params = AstroActivityParams(
                alpha=activity_cfg['alpha'],
                target=activity_cfg['target'])

        elif effect_fn_name == "astro_inc_dec_effect":
            activity_params = AstroActivityParams(
                inc_low_thr = activity_cfg['inc_low_thr'],
                inc_step = activity_cfg['inc_step'],
                max_e = activity_cfg['max'],
                dec_high_thr = activity_cfg['dec_high_thr'],
                dec_step = activity_cfg['dec_step'],
                min_e = activity_cfg['min'])


        else:
            raise ValueError("Unknown effect function {}".format(effect_fn_name))

        params = AstroParams(
            tau=cfg['astro_params']['tau'],
            alpha=cfg['astro_params']['alpha'],
            activity_params=activity_params
        )

        return Astrocyte(params, state_fn, effect_fn, dt=cfg['sim']['dt'])
        

    def _effect(self, state):
        return self.effect_fn(state, self.params)        

    def forward(self, z, state):
        state = self.state_fn(z, self.params, state, dt=self.dt)


        return self._effect(state)


registry.add_entry("astro_state_prop_inc_exp_decay", astro_state_prop_inc_exp_decay)
registry.add_entry("astro_proportional_effect", astro_proportional_effect)
registry.add_entry("astro_const_effect", astro_const_effect)
registry.add_entry("astro_inc_dec_effect", astro_inc_dec_effect)
