import matplotlib.pyplot as plt
from matplotlib import gridspec
import mpld3
from mpld3 import plugins
import pathlib


def figure():
    fig = plt.Figure()
    fig.set_dpi(300)

    return fig


def axis(fig, *args, **kwargs):
    if len(args) > 0:
        if type(args[0]) == gridspec.SubplotSpec:
            ax = fig.add_subplot(args[0])
        else:
            ax = fig.add_subplot(*args[0])
            
            
    else:
        ax = fig.add_subplot()
        
    ax.grid(b=True, which='both')

    hist = 'hist' in kwargs and kwargs['hist']
    if hist:
        bins = kwargs['bins']
        bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
        ax.set_xticks(torch.arange(min(bins)+bin_w/2, max(bins), bin_w), bins)
        ax.set_xlim(bins[0], bins[-1])


    return ax


def save_figure(path, fig):
    path = pathlib.Path(path)
    html_path = path.parent/"{}.html".format(path.name)
    png_path = path.parent/"{}.png".format(path.name)
    fig.savefig(str(png_path))
    
    mpld3.save_html(fig, str(html_path))


def resize_and_gridspec(fig, shape=(5,5), size=(2,2), spacing=0.3):
    fig.set_figheight(shape[1]*size[1]+spacing*(shape[1]+1))
    fig.set_figwidth(shape[0]*size[0]+spacing*(shape[0]+1))

    return gridspec.GridSpec(*shape, wspace=spacing, hspace=spacing)
