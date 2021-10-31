# Torch imports
import torch
from torch import nn

from norse.torch.utils import plot as nplot
from matplotlib import pyplot as plt

# Norse Imports
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFCell
from norse.torch.module.stdp import STDPOptimizer
from norse.eval import logger
from norse.torch.utils import registry, config
from norse.torch.module.astrocyte import Astrocyte

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
                 optimize=True,
                 dt=0.001,
                 dtype=torch.float):

        super(AstroAdaptNet, self).__init__()

        self.cfg = cfg
        dt = cfg['sim']['dt']
        
        self.optimizer = optimizer
        self.optimize = cfg['learning']['optimize']
        
        self.monitors = []

        self.init = False
        self.topology = getattr(self, cfg['topology'])
        self.topology_cfg = cfg.deref('topology')
        self.topology(None)




    def from_cfg(cfg, optimizer):
        return AstroAdaptNet(cfg,
                             optimizer=optimizer,
                             optimize=cfg['learning']['optimize'],
                             dt=cfg['sim']['dt'])


    def adapt_ff_v1(self, z):
        # This code path is taken once
        if not self.init:
            num_inputs = self.topology_cfg['num_inputs']
            self.linear = _linear_from_params(
                num_inputs,
                1,
                self.cfg['linear_params'])
            
            self.lif = (_lif_from_params(self.cfg['lif_params'], dt=0.001), None)
            self.astro = (Astrocyte.from_cfg(self.cfg), None)
            self.last_astro_effect = None
            
            self.init = True
            return

        # Normal forward behavior
        z_pre = z
        z = self.linear(z)

        if not (self.last_astro_effect is None):
            z = z + self.last_astro_effect
        
        z, state = self.lif[0](z, self.lif[1])
        self.lif = (self.lif[0], state)

        effect, state = self.astro[0](z, self.astro[1])
        self.astro = (self.astro[0], state)
        self.last_astro_effect = effect

        return z, self.lif[1], self.astro[1], effect


    def adapt_ff_v2(self, z):
        # This code path is taken once
        if not self.init:
            width = self.topology_cfg['num_inputs']
            self.linear = _linear_from_params(width, 1, self.cfg['linear_params'])
            self.lif = (_lif_from_params(self.cfg['lif_params'], dt=0.001), None)
            self.astro = (Astrocyte.from_cfg(self.cfg), None)
            self.last_astro_effect = None
            
            self.init = True
            return

        # Normal forward behavior
        z_pre = z

        if self.topology_cfg['astro_after_linear']:
            z = self.linear(z)

            effect, state = self.astro[0](z, self.astro[1])
            self.astro = (self.astro[0], state)

        else:
            effect, state = self.astro[0](z, self.astro[1])
            self.astro = (self.astro[0], state)

            z = self.linear(z)

        z += effect

        z, state = self.lif[0](z, self.lif[1])
        self.lif = (self.lif[0], state)
        
        return z, self.lif[1], self.astro[1], effect

    
    def forward(self, z):
        return self.topology(z)

    
def _graph_sim(cfg, monitor):
    fig = plt.Figure(figsize=(10, 15))

    subplot_shape = (6,1)
    
    # Graph timelines
    ax = fig.add_subplot(*subplot_shape,1)
    ax.set_title("Presynaptic Spike Trace")
    ax.set_xlabel("Time")
    ax.set_ylabel("Spikes")
    trace = monitor.trace("pre")
    trace = torch.stack(trace)
    nplot.plot_spikes_2d(trace, axes=ax)

    ax = fig.add_subplot(*subplot_shape,2)
    ax.set_title("Postsynaptic Spike Trace")
    ax.set_xlabel("Time")
    ax.set_ylabel("Spikes")
    trace = monitor.trace("post")
    trace = torch.stack(trace)
    nplot.plot_spikes_2d(trace, axes=ax)

    ax = fig.add_subplot(*subplot_shape,3)
    ax.set_title("Astrocyte State Trace")
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    if monitor.has_trace('astro_state'):
        trace = monitor.trace("astro_state")
        ax.plot(trace, label='State')
    elif monitor.has_trace('astro_state_1'):
        trace = monitor.trace("astro_state_1")
        ax.plot(trace, label='State1')
        trace = monitor.trace("astro_state_2")
        ax.plot(trace, label='State2')
    ax.legend()
    
    ax = fig.add_subplot(*subplot_shape, 4)
    ax.set_title("Astrocyte Effect Trace")
    ax.set_xlabel("Time")
    ax.set_ylabel("Astrocyte Effect")
    trace = monitor.trace("astro_effect")
    ax.plot(trace)

    ax = fig.add_subplot(*subplot_shape, 5)
    ax.set_title("Presynaptic Moving Average")
    ax.set_xlabel("Time")
    ax.set_ylabel("Moving Average Firing Rate")
    trace = monitor.trace("pre_avg_rate", as_numpy=True)
    ax.plot(trace)

    ax = fig.add_subplot(*subplot_shape, 6)
    ax.set_title("Postsynpatic Moving Average")
    ax.set_xlabel("Time")
    ax.set_ylabel("Moving Average Firing Rate")
    trace = monitor.trace("post_avg_rate", as_numpy=True)
    ax.plot(trace)

    fig.subplots_adjust(left=0.1,
                        bottom=0.4, 
                        right=0.9, 
                        top=1.4, 
                        wspace=0.4, 
                        hspace=0.4)

    path = pathlib.Path(logger.get_path())/"{}.png".format(cfg['sim']['sim_name'])
    print("Saving Graphs: ", str(path))
    fig.savefig(str(path), bbox_inches="tight")


def _graph_experiment(cfg, exp_cfg, monitor):

    fig = plt.Figure()

    num_graphs = len(exp_cfg['graphs'])
    subplot_shape = (num_graphs, 1)
    subplot_idx = 1
    
    if 'volume_v_state_alpha' in exp_cfg['graphs']:
        ax = fig.add_subplot(*subplot_shape, subplot_idx)
        subplot_idx += 1

        ax.set_title("Pre and Post Synaptic Volume vs. State Alpha")
        ax.set_xlabel("State Alpha")
        ax.set_ylabel("Spike Volume")
    
        pre_trace = monitor.trace("pre_volume")
        if monitor.has_trace('alpha'):
            state_alpha_trace = monitor.trace("alpha")
        elif monitor.has_trace('alpha1'):
            state_alpha_trace = monitor.trace("alpha1")

        ax.plot(state_alpha_trace, pre_trace, label="Pre")

        post_trace = monitor.trace("post_volume")
        ax.plot(state_alpha_trace, post_trace, label="Post")
        ax.legend()


    if 'range_v_state_alpha' in exp_cfg['graphs']:
        ax = fig.add_subplot(*subplot_shape, subplot_idx)
        subplot_idx += 1

        ax.set_title("Pre and Post Synaptic Moving Average Range vs. State Alpha")
        ax.set_xlabel("State Alpha")
        ax.set_ylabel("Moving Average Range")
    
        pre_trace = monitor.trace("pre_range")
        if monitor.has_trace('alpha'):
            state_alpha_trace = monitor.trace("alpha")
        if monitor.has_trace('alpha1'):
            state_alpha_trace = monitor.trace("alpha1")
            
        ax.plot(state_alpha_trace, pre_trace, label="Pre")

        post_trace = monitor.trace("post_range")
        ax.plot(state_alpha_trace, post_trace, label="Post")
        ax.legend()


    fig.subplots_adjust(left=0.1,
                        bottom=0.4, 
                        right=0.9, 
                        top=1.4, 
                        wspace=0.4, 
                        hspace=0.4)
    
    path = pathlib.Path(logger.get_path())/"{}.png".format(cfg['sim']['exp_name'])

    fig.legend()
    print("Saving Graphs: ", str(path))
    fig.savefig(str(path), bbox_inches="tight")


def _sim(cfg, data, model, monitor, optimizer=None):

    # Clear monitor
    monitor.clear("pre")
    monitor.clear("post")
    monitor.clear("astro_state")
    monitor.clear("astro_effect")
    monitor.clear("weight")
    
    model.optimizer = optimizer
    
    for i, spike_vector in enumerate(data):
        if i > cfg['sim']['iters']:
            break

        z, lif_state, astro_state, astro_effect = model(spike_vector)
        # if optimizer: optimizer.step()

        # Graph a timeline of intermediate steps
        monitor("pre", spike_vector)
        monitor("post", z)
        if 't_z' in astro_state:
            monitor("astro_state", astro_state['t_z'])
        if 't_z1' in astro_state:
            monitor("astro_state_1", astro_state['t_z1'])
            monitor("astro_state_2", astro_state['t_z2'])
        monitor("astro_effect", astro_effect)
        monitor("weight", model.linear.weight.data.view(-1).tolist())

    monitor.moving_average('pre', 'pre_avg_rate', window=50)
    monitor.moving_average('post', 'post_avg_rate', window=50)

    _graph_sim(cfg, monitor)
    
    
def _experiment(cfg, exp_cfg, name=None):
    monitor = logger.TraceLogger()

    print("Experiment: ", name)

    cfg['sim']['exp_name'] = _exp_name_from_cfg(cfg, name)
    
    for i, next_cfg in enumerate(config.iter_configs(cfg, exp_cfg)):
        next_cfg['sim']['sim_name'] = _sim_name_from_cfg(next_cfg, i)

        data = _get_data(next_cfg)
        stdp_optimizer = STDPOptimizer.from_cfg(next_cfg)
        model = AstroAdaptNet.from_cfg(next_cfg, stdp_optimizer)

        print("Running:", next_cfg['sim']['sim_name'])
        _sim(next_cfg, data, model, monitor, optimizer=stdp_optimizer)

        # Aggregate stats
        pre_trace = monitor.trace('pre', as_tensor=True)
        post_trace = monitor.trace('post', as_tensor=True)
        monitor('post_volume', post_trace.sum())
        monitor('pre_volume', pre_trace.sum())

        pre_trace = monitor.trace('pre_avg_rate', as_tensor=True)
        post_trace = monitor.trace('post_avg_rate', as_tensor=True)
        monitor('pre_range', pre_trace.max() - pre_trace.min())
        monitor('post_range', post_trace.max() - post_trace.min())

        monitor('rate', next_cfg['data']['stops'])
        
        if 'tau' in next_cfg['astro_params']:
            monitor('tau', next_cfg['astro_params']['tau'])

        if 'alpha' in next_cfg['astro_params']:
            monitor('alpha', next_cfg['astro_params']['alpha'])
        elif 'alpha1' in next_cfg['astro_params']:
            monitor('alpha1', next_cfg['astro_params']['alpha1'])
            monitor('alpha2', next_cfg['astro_params']['alpha2'])

    _graph_experiment(cfg, exp_cfg, monitor)

        
def _gen_poisson_ramp(cfg):

    width = cfg['data']['width']
    stops = cfg['data']['stops']
    steps = cfg['data']['steps']

    if not hasattr(stops, '__iter__'):
        stops = [stops]
    if len(stops) == 0:
        raise ValueError("At least one stop is needed for poisson ramp")
    if not hasattr(stops[0], '__iter__'):
        stops = [stops]
        
    # stops should be [[seq for out 1], [seq for out 2], ...]

    stops = torch.Tensor(stops)
    rate_seqs = []
    for i, seq in enumerate(stops):
        # If there is only one rate, pass it along
        if len(seq) == 1:
            rate_seqs.append(torch.Tensor(seq))
            continue

        # Else, expand the stops into a sequence of rates
        seq_exp = torch.Tensor([])
        steps_per_stop = int(steps / (len(seq) - 1))
        
        for i in range(len(seq) - 1):
            start = float(seq[i])
            end = float(seq[i+1])
            seq_exp = torch.cat((seq_exp, torch.linspace(start, end, steps_per_stop)))

        rate_seqs.append(seq_exp)

    # Transpose so axis 0 is iters
    rate_seq = torch.stack(rate_seqs).transpose(1,0)
    
    while True:
        for rate in rate_seq:
            spike_vector = torch.poisson(rate).view(width)
            
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


def _sim_name_from_cfg(cfg, i):    
    astro_config = cfg['astro_params']
    effect_config = astro_config.deref('effect_params')

    name = cfg['sim']['exp_name'] + "_"

    if 'alpha' in astro_config:
        try:
            name += "alpha{:07.3f}_".format(astro_config['alpha'])
        except:
            import code
            code.interact(local=dict(globals(), **locals()))
            exit(1)
    if 'tau' in astro_config:
        name += "tau{:07.3f}_".format(astro_config['alpha'])
    
    if 'target' in effect_config:
        name += "tar{:07.3f}_".format(effect_config['target'])
    elif 'alpha' in effect_config:
        name += "eal{:07.3f}"

    name += "{:04d}".format(i)

    return name


def _exp_name_from_cfg(cfg, exp_name):
    return exp_name
        

def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str)
    
    return parser.parse_args()


def _main():
    args = _parse_args()

    logger.set_path('./runs/astro_adapt')

    cfg = config.load_config(args.config)

    with torch.no_grad():
        for exp_name in cfg['meta']['experiments']:
            exp_cfg = cfg['meta'][exp_name]
            _experiment(cfg, exp_cfg, name=exp_name)
        
        
if __name__ == '__main__':
    _main()
