import argparse
import torch
import numpy as np
import os
from matplotlib import pyplot as plt
from norse.torch.utils.plot import plot_spikes_2d, plot_heatmap_2d

from norse.torch.module import stdp
from norse.torch.functional.lif import (
    LIFFeedForwardState,
    LIFParameters,
)
from norse.torch.module.lif import LIFCell

from norse.torch.module import encode

from norse.eval import graph
import pathlib

def _graph_lif_trace(z_trace, state_trace, param_descr, dt=0.001):
    fig = plt.Figure()
    fig.suptitle(param_descr)
    ax = fig.add_subplot(311)

    # Check shapes
    if not (len(z_trace.shape) in [1, 2]): raise ValueError("z_trace must be 1-d or 2-d")
    if len(z_trace.shape) == 1:
        z_trace = z_trace.view(z_trace.shape[0], -1)
    
    # Plot z_trace
    ax = plot_heatmap_2d(z_trace, ax)
    ax.set_xticks(np.arange(0, z_trace.shape[0], 50))

    # Plot state voltage and current
    i = [s.i.detach().cpu().numpy() for s in state_trace]
    i = np.vstack(i)
    v = [s.v.detach().cpu().numpy() for s in state_trace]
    v = np.vstack(v)

    t = [i * dt for i in range(len(v))]

    ax = fig.add_subplot(312)
    ax.set_title("LIF Neuron Voltage vs. Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Membrane Voltage")
    
    ax.set_xticks(np.arange(0, max(v), 50))
    ax.plot(t, v)

    ax = fig.add_subplot(313)
    ax.set_title("LIF Neuron Input Current vs. Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Input Current")
    ax.set_xticks(np.arange(0, max(i), 50))
    ax.plot(t, i)

    fig.tight_layout()

    return fig


def _get_lif_params():
    v_th = 10.0
    v_reset = -v_th/2.0
    
    for tau_syn_inv in np.linspace(150, 250.0, num=10):
        yield LIFParameters(tau_mem_inv=100.0, tau_syn_inv=tau_syn_inv, v_th=v_th, v_reset=v_reset), "tau_syn_inv-{:04.2f}".format(tau_syn_inv)

    for tau_mem_inv in np.linspace(50.0, 150.0, num=10):
        yield LIFParameters(tau_mem_inv=tau_mem_inv, v_th=v_th, v_reset=v_reset), "tau_mem_inv-{:04.2f}".format(tau_mem_inv)


def _inspect_lif(path):

    path = pathlib.Path(path)
    if not path.exists():
        os.mkdir(str(path))

    for lif_params, params_descr in _get_lif_params():
        lif_n = LIFCell(p=lif_params)
        lif_state = None
        z_trace = []
        state_trace = []

        print("evaluating {}".format(params_descr))
        for i in range(100):
            if i < 20:
                v = torch.Tensor([0.2])
            else:
                v = torch.Tensor([0.0])

            z, lif_state = lif_n(v, lif_state)
            z_trace.append(z)
            state_trace.append(lif_state)

        z_trace = torch.vstack(z_trace)
        
        fig = _graph_lif_trace(z_trace, state_trace, params_descr)

        fig.savefig(str(path/"lif_trace_{}.png".format(params_descr)))


def spike_deltas(spikes):
    spikes = spikes.view(spikes.shape[0], -1)

    spikes = torch.transpose(spikes, 0, 1)

    spike_deltas = []
    
    for neuron in torch.unbind(spikes):
        neuron = torch.nonzero(neuron)
        spike_delta = neuron[1:] - neuron[:-1]
        spike_deltas.append(spike_delta)

    return spike_deltas


def _inspect_stdp(path):
    fig = graph.figure()
    for i, algo in enumerate(['additive', 'additive_step', 'multiplicative_pow', 'multiplicative_relu']):
        dt, dw = stdp.inspect(algo=algo, steps=1000)

        ax = graph.axis(fig, (2,2,i+1))
    
        ax.set_title("Pre-to-Post Delay vs. STDP weight update: {}".format(algo))
        ax.set_xlabel("Pre-to-Post Delay")
        ax.set_ylabel("Weight Update")
        ax.plot(dt.cpu().numpy(), dw.cpu().numpy())
    
    graph.save_figure(path, fig)


def _inspect_encode(path):

    p = LIFParameters()
    seq_length = 10
    pop_out_features = 5
    path = pathlib.Path(path)
    
    encoders = [
        encode.ConstantCurrentLIFEncoder(seq_length, p),
        encode.PoissonEncoder(seq_length),
        encode.PopulationEncoder(pop_out_features),
        encode.SignedPoissonEncoder(seq_length),
        encode.SpikeLatencyLIFEncoder(seq_length, p),
        encode.SpikeLatencyEncoder(),
    ]

    x = torch.rand(5)

    fig = plt.figure(tight_layout=True)
    fig.suptitle("Spike Encoding For: {}".format(x))

    subplot_shape = (np.ceil(np.sqrt(len(encoders))), np.floor(np.sqrt(len(encoders))))
    
    for i, encoder in enumerate(encoders):
        ax = fig.add_subplot(*subplot_shape, i+1)

        ax.set_title(str(encoder))
        
        print("Testing: ", encoder, end='')
        enc = encoder(x)
        print(" - ", enc.shape)

        if len(enc.shape) == 1:
            enc = enc.view(-1, enc.shape[0])

        plot_spikes_2d(enc, ax)

    fig.savefig(str(path/"spike_encode.png"))


def _inspect_encode_mnist(path):
    from norse.torch.module.encode import ConstantCurrentLIFEncoder
    import torch.utils.data
    import torchvision
    
    constant_current_encoder = ConstantCurrentLIFEncoder(seq_length=100)
    
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root=".",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    # torchvision.transforms.
                    #    RandomCrop(size=[28,28], padding=4)
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=1,
        shuffle=True
    )

    sample_spikes = []
    sampled_classes = []

    for batch_idx, (data, target) in enumerate(train_loader):
        x = constant_current_encoder(
            data.view(-1, 28*28) * 1.0
        )

        if target.numpy()[0] in sampled_classes:
            continue

        sample_spikes.append((target.numpy()[0], x.squeeze()))
        sampled_classes.append(target.numpy()[0])

        if len(sample_spikes) == 10:
            break


    # Spikes Event Plot
    fig = graph.figure()
    gs = graph.resize_and_gridspec(fig, shape=(5,5), size=(3,3))

    for label, spikes in sample_spikes:
        ax = graph.axis(fig, gs[label])
        ax.set_title("Spike train for Class {}".format(label))
        ax.set_xlabel("Time")
        ax.set_ylabel("Neuron")

        spike_idx = []
        for i in range(spikes.shape[1]):
            neuron = spikes[:, i]
            neuron_spikes = torch.nonzero(neuron).view(-1).numpy().tolist()
            spike_idx.append(neuron_spikes)

        ax.eventplot(spike_idx)

    graph.save_figure("{}_spikes".format(path), fig)

    # Spike Delta plot
    fig = graph.figure()
    ax = graph.axis(fig)

    ax.set_title("Range of spike deltas per neuron")
    ax.set_xlabel("Time simulation time (dt units)")
    ax.set_ylabel("Spike delta time (dt units)")

    for neuron in spike_idx:
        spike_delta = torch.Tensor(neuron)
        spike_delta = spike_delta[1:] - spike_delta[:-1]
        
        if len(torch.nonzero(spike_delta)) == 0:
            continue

        spike_delta = spike_delta * constant_current_encoder.dt

        ax.plot(spike_delta.numpy().tolist(), '-')
    
    graph.save_figure("{}_spike_delta".format(path), fig)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lif', action='store_true')
    parser.add_argument('--stdp', action='store_true')
    parser.add_argument('--encode', action='store_true')
    
    parser.add_argument('--output', '-o', type=str)

    return parser.parse_args()


def _main():
    args = _parse_args()

    path = pathlib.Path(args.output)
    if not path.exists():
        raise ValueError("--output must exist")

    if args.stdp:
        path_stdp = path/"stdp"
        _inspect_stdp(str(path_stdp))

    if args.lif:
        path_lif = path/"lif"
        _inspect_lif(str(path_lif))

    if args.encode:
        path_enc = path/"encode"
        _inspect_encode(str(path_enc))


    # _inspect_encode_mnist(str(path_encode))


if __name__ == '__main__':
    _main()
