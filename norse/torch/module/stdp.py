import torch

from norse.torch.module.lif import LIFCell
from norse.torch.functional.stdp import stdp_step_linear, stdp_step_conv2d, STDPState, STDPParameters


class STDPOptimizer:
    def __init__(self):
        # Setup stdp objects in case forward is called with stdp=true
        self.stdp_conv_params = STDPParameters(
            eta_minus=1e-4,
            eta_plus=1e-4,
            hardbound=True,
            w_min=-1.0, w_max=1.0,
            convolutional=True
        )

        self.stdp_lin_params = STDPParameters(
            eta_minus=1e-4,
            eta_plus=1e-4,
            hardbound=True,
            w_min=-1.0, w_max=1.0,
        )

        self.stdp_steps = []


    def to(self, device):
        self.stdp_conv_params.to(device)
        self.stdp_lin_params.to(device)


    def init_stdp(self, module, in_x, out_x):
        if not hasattr(module, "stdp_state"):
            print("Allocating STDP state for: ", module)
            module.stdp_state = STDPState(
                t_pre=torch.zeros(in_x.shape).to(in_x.device),
                t_post=torch.zeros(out_x.shape).to(in_x.device)
            )


    def __call__(self, module, z_pre, z):
        self.stdp_steps.append((module, z_pre, z))


    def _stdp_step(self, module, z_pre, z, reward=None):                      
        def _is_conv(obj):
            return type(obj) == torch.nn.Conv2d
    
        def _is_linear(obj):
            return type(obj) == torch.nn.Linear

        self.init_stdp(module, z_pre, z)
        
        alloc_before = torch.cuda.memory_allocated()
        
        w0 = module.weight.detach()

        # if reward is none, let stdp_step_* modify the weights, otherwise no
        if reward is None:
            w = w0
        else:
            w = torch.zeros_like(w0)

        stdp_state = module.stdp_state
        if _is_conv(module):
            w, stdp_state, dw = stdp_step_conv2d(
                z_pre, z, w,
                stdp_state,
                self.stdp_conv_params,
                dt=0.001
            )
        elif _is_linear(module):
            w, stdp_state, dw = stdp_step_linear(
                z_pre, z, w,
                stdp_state,
                self.stdp_lin_params,
                dt=0.001
            )

        # If there is a reward, apply it with dw
        if not (reward is None):
            w = w0 + (dw * reward)
            
        module.weight.data = w

        if not hasattr(module, 'avg_dw'):
            module.avg_dw = 0.0
        module.avg_dw = (module.avg_dw + torch.mean(dw)) / 2


    def step(self):
        for step in self.stdp_steps:
            self._stdp_step(*step)

        self.stdp_steps = []


    def step_reward(self, reward):
        with torch.no_grad():
            for step in self.stdp_steps:
                self._stdp_step(*step, reward=reward)

        self.stdp_steps = []
