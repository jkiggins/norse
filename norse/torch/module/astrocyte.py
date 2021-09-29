"""
NN Module that models (to a degree) astrocyte behavior
"""

from norse.torch.functional.astrocyte import AstroParams, AstroActivityParams, astro_step

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


    def effect(state):
        pass
        

    def forward(pre_spikes, post_spikes, state):
        self.astro_step(pre_spikes, post_spikes, params)
        if self.mod_activity:
            astrocyte_step_activity(state, self.params, module)

        return state
        
