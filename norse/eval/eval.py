import argparse

import torch
import norse

import matplotlib.pyplot as plt, mpld3


def _graph_params(params, title):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_title(title)
    ax.set_xlabel("Bins")
    ax.set_ylabel("Counts")

    num_bins = 100
    params = params.cuda()
    hist_counts = torch.histc(params, bins=num_bins, min=params.min(), max=params.max()).cpu()
    hist_step = (params.max() - params.min()) / num_bins
    hist_bins = torch.Tensor([params.min() + hist_step * i for i in range(num_bins)])

    ax.plot(hist_bins, hist_counts)

    return fig
    


def _load_state_dict(path):
    state_dict = torch.load(path)

    return state_dict['model_state_dict']

    
def _get_flat_params(state_dict):
    conv_params=torch.Tensor()
    lin_params=torch.Tensor()

    for key in state_dict:
        params = state_dict[key]

        if not 'weight' in key:
            continue

        if len(params.shape) == 2:
            lin_params=torch.cat((lin_params, params.cpu().view(-1)))
        else:
            conv_params=torch.cat((conv_params, params.cpu().view(-1)))


    return lin_params, conv_params


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--state-dict", type=str, required=True)

    return parser.parse_args()
    
    
def _main():
    args = _parse_args()

    state_dict = _load_state_dict(args.state_dict)
    lin_params, conv_params = _get_flat_params(state_dict)

    lin_fig = _graph_params(lin_params, "Histogram of Linear Weight Values")
    conv_fig = _graph_params(conv_params, "Histogram of Conv Weight Values")

    mpld3.save_html(lin_fig, "linear_histogram.html")

    
if __name__ == '__main__':
    _main()
