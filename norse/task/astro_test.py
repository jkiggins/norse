# Torch imports
import torch
from torch import nn

# Norse Imports
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFCell
from norse.torch.module.stdp import STDPOptimizer
from norse.eval import logger
from norse.torch.utils import registry, config
from norse.torch.module.astrocyte import Astrocyte
from norse.torch.functional.astrocyte import AstroState, AstroParams, AstroActivityParams

# Other Libs
from matplotlib import pyplot as plt

from norse.eval import logger

import pathlib
import argparse

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
                 lif_params,
                 linear_params,
                 optimizer=None,
                 optimize=True,
                 topology=None,
                 dt=0.001,
                 dtype=torch.float):

        super(AstroAdaptNet, self).__init__()

        dt = cfg['sim']['dt']
        
        self.optimizer = optimizer
        self.optimize = cfg['learning']['optimize']
        
        self.monitors = []

        top = topology

        self.linear = _linear_from_params(top['input']['num'], top['lif']['num'], linear_params)
        self.lif = _lif_from_params(lif_params)
        self.lif_state = None

        self.astro = Astrocyte.from_cfg(cfg)
        self.astro_state = None
        self.last_astro_effect = None


    def from_cfg(cfg, optimizer):
        top_key = cfg['topology']

        if not (top_key in cfg):
            raise ValueError("Topology {} not defined in config".format(top_key))
        top = cfg[top_key]

        return AstroAdaptNet(cfg,
                             lif_params=cfg['lif_params'],
                             linear_params=cfg['linear_params'],
                             optimizer=optimizer,
                             optimize=cfg['learning']['optimize'],
                             topology=top,
                             dt=cfg['sim']['dt'])


    def forward(self, z):
        z_pre = z

        z = self.linear(z)
        if self.last_astro_effect: z += self.last_astro_effect
        z, lif_state = self.lif(z, self.lif_state)

        effect, astro_state = self.astro.forward(z, self.astro_state)
        
        self.last_astro_effect = effect
        
        self.astro_state = astro_state
        self.lif_state = lif_state

        if self.optimize: self.optimizer(self.linear, z_pre, z)

        return z, lif_state, astro_state, effect


def _sim(cfg, data, model, monitor, optimizer=None, name=None):

    model.optimizer = optimizer
    
    for i, spike_vector in enumerate(data):
        if i > cfg['sim']['iters']:
            break

        z, lif_state, astro_state, astro_effect = model(spike_vector)
        if optimizer: optimizer.step()

        # Graph a timeline of intermediate steps
        monitor("pre", spike_vector)
        monitor("post", z)
        monitor("astro_state", astro_state.t_z)
        monitor("astro_effect", astro_effect)
        monitor("weight", float(model.linear.weight.data.view(-1)))

    monitor.graph('pre', '2d_spike_plot')
    monitor.graph('post', '2d_spike_plot')
    monitor.graph('astro_state', 'scalar')
    monitor.graph('astro_effect', 'scalar')

    monitor.moving_average('pre', 'pre_avg_rate', window=50)
    monitor.moving_average('post', 'post_avg_rate', window=50)

    monitor.graph('pre_avg_rate', 'scalar')
    monitor.graph('post_avg_rate', 'scalar')
    monitor.graph('weight', 'scalar')

    # Compute aggregate stats for later graphing
    activity_config = cfg['astro_params'].deref('activity_params')
    pre_trace = monitor.trace('pre', as_tensor=True).mean()
    monitor("pre_avg", pre_trace, x=activity_config['target'])
    post_trace = monitor.trace('post', as_tensor=True).mean()
    monitor("post_avg", post_trace, x=activity_config['target'])


    fig = monitor.figure(clear_traces=True)
    path = pathlib.Path(logger.get_path())/"{}.png".format(name)
    fig.savefig(str(path))

    
def _experiment(cfg, exp_cfg, name=None):
    monitor = logger.TraceLogger()
    
    for i, next_cfg in enumerate(config.iter_configs(cfg, exp_cfg)):
        sim_name_fn = registry.get_entry(cfg['sim']['name_fn'])
        exp_name_fn = registry.get_entry(exp_cfg['name_fn'])
        cfg['sim']['name'] = sim_name_fn(cfg)
        exp_name = exp_name_fn(cfg, name)
        
        pprint(next_cfg)

        data = _get_data(next_cfg)
        stdp_optimizer = STDPOptimizer.from_cfg(next_cfg)
        model = AstroAdaptNet.from_cfg(next_cfg, stdp_optimizer)

        _sim(cfg, data, model, monitor, optimizer=stdp_optimizer, name="{}.{:04d}".format(exp_name, i))

    monitor.graph('pre_avg', 'scalar')
    monitor.graph('post_avg', 'scalar')

    fig, axes = monitor.figure(return_axes=True)
    axes['pre_avg'].set_title("Presynaptic Average Firing Rate vs. Astrocyte State Target")
    axes['pre_avg'].set_xlabel("Astrocyte State Target")
    axes['pre_avg'].set_ylabel("Average pre-synaptic Spiking Activity")

    axes['post_avg'].set_title("Post-synaptic Average Firing Rate vs. Astrocyte State Target")
    axes['post_avg'].set_xlabel("Astrocyte State Target")
    axes['post_avg'].set_ylabel("Average post-synaptic Spiking Activity")

    path = pathlib.Path(logger.get_path())/"{}.aggregate.png".format(name)
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


def _name_from_cfg(cfg):
    activity_config = cfg['astro_params'].deref('activity_params')
    target = activity_config['target']

    astro_tau = cfg['astro_params']['tau']
    astro_alpha = cfg['astro_params']['alpha']
    
    name = "a{:05.3f}_t{:05.3f}_tar{:05.3f}".format(astro_alpha, astro_tau, target)

    return name


def _exp_name_from_cfg(cfg, exp_name):
    name = "{}_{}".format(exp_name, cfg['sim']['name'])

    return name


def _fill_registry():
    registry.add_entry('_name_from_cfg', _name_from_cfg)
    registry.add_entry('_exp_name_from_cfg', _exp_name_from_cfg)
        

def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str)
    
    return parser.parse_args()



def _main():
    args = _parse_args()

    logger.set_path('./runs/astro_test')
    _fill_registry()

    cfg = config.load_config(args.config)
    
    with torch.no_grad():
        for exp_name in cfg['meta']['experiments']:
            _experiment(cfg, cfg['meta'][exp_name], name=exp_name)
        
        
if __name__ == '__main__':
    _main()
