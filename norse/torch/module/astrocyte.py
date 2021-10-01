"""
NN Module that models (to a degree) astrocyte behavior
"""

from norse.torch.functional.astrocyte import AstroParams, AstroActivityParams, astro_step, astro_get_presynaptic_current

class Astrocyte:
    def __init__(self, params, mod_learning=False, mod_activity=False):
        self.params = params
        self.mod_learning = mod_learning
        self.mod_activity = mod_activity
        self.astro_step = astro_step


    def from_cfg(cfg):

        learning_params = None
        if cfg['adapt']['enabled']:
            learning_params = AstroActivityParams(target=cfg['adapt']['target'])
                                                  
        params = AstroParams(alpha=cfg['alpha'],
                                 tau=cfg['tau'],
                                 learning_params=learning_params)

        return Astrocyte(params, mod_activity=cfg['adapt']['enabled'])


    def _effect(state):
        effect_dict = {
            'post_synaptic_current': astro_get_presynaptic_current(state)
        }

        return effect_dict
        

    def forward(pre_spikes, post_spikes, state):
        state = self.astro_step(pre_spikes, post_spikes, self.params)

        return self._effect(state), state
        
