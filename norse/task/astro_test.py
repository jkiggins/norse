# Torch imports
import torch
from torch import nn

# Norse Imports
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFCell
from norse.torch.module.stdp import STDPOptimizer
from norse.eval import logger
from norse.torch.utils import registry
from norse.torch.module.astrocyte import Astrocyte
from norse.torch.functional.astrocyte import AstroState, AstroParams, AstroActivityParams

# Other Libs
from matplotlib import pyplot as plt

from norse.eval import logger

import pathlib
import argparse
import oyaml as yaml
from pprint import pprint


def _lif_from_params(cfg, dt=0.001):
    lif_params = LIFParameters(method=cfg['method'], v_reset=cfg['v_reset'], v_th=cfg['v_th'])
    
    return LIFCell(p=lif_params, dt=dt)


def _linear_from_params(in_features, out_features, params):
    layer = nn.Linear(in_features, out_features, bias=params['bias'])
    
    if params['init'][0] == 'uniform':
        nn.init.uniform_(layer.weight)
    elif params['init'][0] == 'unity':
        print("init weights unity")
        layer.weight = nn.Parameter(torch.ones_like(layer.weight))
    elif params['init'][0] == 'normal':
        mu = params['init'][1]
        sigma = params['init'][2]
        nn.init.normal_(layer.weight, mean=mu, std=sigma)
                    
    torch.clip(layer.weight, 0.0, 1.0)

    return layer



class AstroAdaptNet(torch.nn.Module):
    # TODO: Convert to from_cfg format
    def __init__(self, cfg,
                 optimizer=None,
                 dtype=torch.float):

        super(AstroAdaptNet, self).__init__()

        dt = cfg['sim']['dt']
        
        self.optimizer = optimizer
        self.monitors = []

        top_key = cfg['topology']

        if not (top_key in cfg):
            raise ValueError("Topology {} not defined in config".format(top_key))
        top = cfg[top_key]

        self.linear = _linear_from_params(top['input']['num'], top['lif']['num'], cfg['linear_params'])
        self.lif = _lif_from_params(cfg['lif_params'])
        self.lif_state = None

        self.astro = Astrocyte.from_cfg(cfg)
        self.astro_state = None
        self.last_astro_effect = None


    def forward(self, z):
        z_pre = z

        z = self.linear(z)
        if self.last_astro_effect: z += self.last_astro_effect
        z, lif_state = self.lif(z, self.lif_state)

        effect, astro_state = self.astro.forward(z, self.astro_state)
        self.last_astro_effect = effect
        
        self.astro_state = astro_state
        self.lif_state = lif_state

        return z, lif_state, astro_state, effect


def _sim(cfg, data, model, monitor):
    
    for i, spike_vector in enumerate(data):
        if i > cfg['sim']['iters']:
            break

        z, lif_state, astro_state, astro_effect = model(spike_vector)

        # Graph a timeline of intermediate steps
        monitor("pre", spike_vector)
        monitor("post", z)
        monitor("astro_state", astro_state.t_z)
        monitor("astro_effect", astro_effect)

    monitor.graph('pre', '2d_spike_plot')
    monitor.graph('post', '2d_spike_plot')
    monitor.graph('astro_state', 'scalar')
    monitor.graph('astro_effect', 'scalar')

    monitor.moving_average('pre', 'pre_avg_rate', window=50)
    monitor.moving_average('post', 'post_avg_rate', window=50)

    monitor.graph('pre_avg_rate', 'scalar')
    monitor.graph('post_avg_rate', 'scalar')

    fig = monitor.figure()
    path = pathlib.Path(logger.get_path())/"{}.png".format(cfg['sim']['name'])
    fig.savefig(str(path))


def _gen_poisson_ramp(cfg):

    width = cfg['data']['width']
    stops = cfg['data']['stops']
    steps = cfg['data']['steps']

    if len(stops) == 0:
        raise ValueError("At least one stop is needed for poisson ramp")
    elif len(stops) == 1:
        rate_seq = stop
    else:
        rate_seq = torch.Tensor([])
        steps_per_stop = int(steps / (len(stops) - 1))
        
        for i in range(len(stops)-1):
            start = stops[i]
            stop = stops[i+1]
            
            rate_seq = torch.cat((rate_seq, torch.linspace(start, stop, steps_per_stop+1)))

    while True:
        for rate in rate_seq:
            p_rate = torch.ones((width)) * rate

            spike_vector = torch.poisson(torch.as_tensor(p_rate)).view(1)
            
            yield (spike_vector > 0.1)*1.0
    

def _get_data(cfg):
    # Handle generated data
    def _get_gen_data(cfg):
        gen_data_functions = {
            'poisson_ramp': _gen_poisson_ramp,
        }

        return gen_data_functions[cfg['data']['gen']](cfg)

    if 'gen' in cfg['data']:
        return _get_gen_data(cfg)

    
def _load_config(path):
    path = pathlib.Path(path)

    if not path.exists():
        raise ValueError("Path: {} doesn't exist".format(str(path)))

    with open(str(path), 'rb') as f:
        cfg = yaml.safe_load(f)

    return cfg


def _meta_apply_overlay(overlay, cfg):
    # Helper functions
    def _apply_dict(overlay, cfg, next_arr):
        for k, v in overlay.items():
            if k in cfg and type(cfg[k]) == dict:
                next_arr.append((v, cfg[k]))
            elif k in cfg:
                cfg[k] = v
                
    apply_next = [(overlay, cfg)]

    while len(apply_next) > 0:
        overlay, cfg = apply_next.pop(0)
        
        if (type(overlay) != dict) or (type(cfg) != dict):
            raise ValueError("Error appplying {} over {}".format(type(overlay), type(cfg)))

        if type(overlay) == dict:
            _apply_dict(overlay, cfg, apply_next)
        


def _model_and_cfg_from_meta(cfg):
    if not ('meta' in cfg) or len(cfg['meta']['experiments']) == 0:
        yield cfg, AstroAdaptNet(cfg, optimizer=None)
        return

    for experiment in cfg['meta']['experiments']:
        if not (experiment in cfg['meta']):
            raise ValueError("Experiment {} not found in meta".format(experiment))
        exp_cfg = cfg['meta'][experiment]

        if 'overlay' in exp_cfg:
            _meta_apply_overlay(exp_cfg['overlay'], cfg)
        
        if 'vary_astro_target' in exp_cfg:
            low, high, steps = exp_cfg['vary_astro_target']
            astro_targets = torch.linspace(low, high, steps+1)

            sim_base_name = cfg['sim']['name']

            activity_cfg_name = cfg['astro_params']['activity_params']
            activity_cfg = cfg['astro_params'][activity_cfg_name]
            for target in astro_targets:
                activity_cfg['target'] = target
                cfg['sim']['name'] = "{}_{:04.1f}".format(sim_base_name, target)

                model = AstroAdaptNet(cfg, optimizer=None)

                yield cfg, model
        else:
            raise ValueError("Experiment not formatted properly")


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str)
    
    return parser.parse_args()


def _main():
    args = _parse_args()

    logger.set_path('./runs/astro_test')

    cfg = _load_config(args.config)
    data = _get_data(cfg)

    aggregate_monitor = logger.NeuroMonitor()

    with torch.no_grad():

        for cfg, model in _model_and_cfg_from_meta(cfg):
            monitor = logger.NeuroMonitor()
            _sim(cfg, data, model, monitor)

            # Compute aggregate stats
            # pre = monitor.trace('pre')
            # post = monitor.graph('post')
            # astro_state = monitor.graph('astro_state')
            # astro_effect = monitor.graph('astro_effect')

            


if __name__ == '__main__':
    _main()
