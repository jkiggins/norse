import argparse
import torch
import numpy as np
import matplotlib
from norse.torch.module import stdp
from . import graph
import pathlib


def spike_deltas(spikes):
    spikes = spikes.view(spikes.shape[0], -1)

    spikes = torch.transpose(spikes, 0, 1)

    spike_deltas = []
    
    for neuron in torch.unbind(spikes):
        neuron = torch.nonzero(neuron)
        spike_delta = neuron[1:] - neuron[:-1]
        spike_deltas.append(spike_delta)

    return spike_deltas


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", "-o", type=str)

    return parser.parse_args()


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


    
def _main():
    args = _parse_args()

    path = pathlib.Path(args.output)
    if not path.exists():
        raise ValueError("--output must exist")

    path_stdp = path
    if path_stdp.is_dir():
        path_stdp = path_stdp/"inspect_stdp"
    _inspect_stdp(str(path_stdp))

    path_encode = path
    if path_encode.is_dir():
        path_encode = path_encode/"inspect_encode"
    _inspect_encode_mnist(str(path_encode))


if __name__ == '__main__':
    _main()
