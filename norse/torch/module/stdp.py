import torch

from norse.torch.module.lif import LIFCell
from norse.torch.functional.stdp import stdp_step_linear, stdp_step_conv2d, STDPState, STDPParameters

class STDPOptimizer:
    def __init__(self):
        # Setup stdp objects in case forward is called with stdp=true
        self.stdp_conv_params = STDPParameters(
            eta_minus=1e-3,
            eta_plus=1e-3,
            hardbound=True,
            w_min=-1.0, w_max=1.0,
            convolutional=True
        )

        self.stdp_lin_params = STDPParameters(
            eta_minus=1e-3,
            eta_plus=1e-3,
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

        return dw
    

    def step(self):
        dw_arr = []
        for step in self.stdp_steps:
            dw = self._stdp_step(*step)
            dw_arr.append(dw.mean())

        self.stdp_steps = []

        return torch.Tensor(dw_arr)


    def step_reward(self, reward):
        with torch.no_grad():
            for step in self.stdp_steps:
                self._stdp_step(*step, reward=reward)

        self.stdp_steps = []


def inspect(algo="additive", steps=100):
    n_batches = 1
    n_pre = 1
    n_post = 1
    dt = 0.001
    mu = 0.0
    n_sweep = steps

    w = torch.Tensor([1.0]).view(1,1)
    
    p_stdp = STDPParameters(
        eta_minus=1e-1,
        eta_plus=3e-1,  # Best to check with large, asymmetric learning-rates
        stdp_algorithm=algo,
        mu=mu,
        hardbound=False,
        convolutional=False,
    )

    dw_arr = []
    dt_arr = []
    
    for i in range(n_sweep):
        spike_times = [i, n_sweep//2]
        start = min(spike_times)
        stop = max(spike_times)

        # Reset the state for reach run
        state_stdp = STDPState(
            t_pre=torch.zeros(n_batches, n_pre),
            t_post=torch.zeros(n_batches, n_post),
        )

        dt_arr.append((spike_times[1] - spike_times[0]) * dt)

        for t in range(start, stop+1):
            z_pre = torch.Tensor([t == spike_times[0]]).view(1,1)
            z_post = torch.Tensor([t == spike_times[1]]).view(1,1)
            
            w, state_stdp, dw = stdp_step_linear(
                z_pre,
                z_post,
                w,
                state_stdp,
                p_stdp,
                dt=dt,
            )

            if t != stop:
                assert dw < 1e-3

        dw_arr.append(dw)

    return torch.Tensor(dt_arr), torch.Tensor(dw_arr)
                
