"""
In this task, we train a spiking convolutional network to learn the
MNIST digit recognition task.
"""
from argparse import ArgumentParser
import os
import uuid

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torchvision

from norse.torch.models.conv import ConvNetStdp
from norse.torch.module.stdp import STDPOptimizer
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.eval import logger


class LIFConvNet(torch.nn.Module):
    def __init__(
        self,
        input_features,
        seq_length,
        input_scale,
        model="super",
        only_first_spike=False,
        optimizer=None,
    ):
        super(LIFConvNet, self).__init__()
        self.constant_current_encoder = ConstantCurrentLIFEncoder(seq_length=seq_length)
        self.only_first_spike = only_first_spike
        self.input_features = input_features
        self.rsnn = ConvNetStdp(method=model, optimizer=optimizer)
        self.seq_length = seq_length
        self.input_scale = input_scale
        self.optimizer = optimizer


    def to(self, device):
        super(LIFConvNet, self).to(device)

        self.rsnn = self.rsnn.to(device)
        self.optimizer.to(device)

        return self


    def no_grad(self):
        for p in self.parameters():
            p.requires_grad = False

        self.rsnn.no_grad()


    def forward(self, x, optimize=True):
        batch_size = x.shape[0]

        # Add time dimension using some encoder
        x = self.constant_current_encoder(
            x.view(-1, self.input_features) * self.input_scale
        )

        x = x.reshape(self.seq_length, batch_size, 1, 28, 28)
        voltages = self.rsnn(x, optimize=optimize)
        m, _ = torch.max(voltages, 0)
        log_p_y = torch.nn.functional.log_softmax(m, dim=1)
            
        return log_p_y


def train(
    model,
    optimizer,
    device,
    train_loader,
    epoch,
    clip_grad,
    grad_clip_value,
    epochs,
    log_interval,
    do_plot,
    plot_interval,
    seq_length,
    writer,
):
    # import code
    # code.interact(local=dict(globals(), **locals()))
    # exit(1)
        
    losses = []

    batch_len = len(train_loader)
    step = batch_len * epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # data is "normal" here, no time dimension
        
        output = model(data).detach()
        output_classes = torch.argmax(output, axis=1)

        dw_arr = optimizer.step()
        
        num_correct = torch.sum(output_classes == target)
        accuracy = num_correct / len(output_classes)

        vote = ((accuracy > 0.5) * 2) - 1
        optimizer.step_reward(vote)

        step += 1

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    epochs,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

            trace_logger = logger.TraceLogger()
            trace_logger.set_trace("dw", dw_arr)

            trace_logger.log_traces()

        if step % log_interval == 0:
            _, argmax = torch.max(output, 1)
            accuracy = (target == argmax.squeeze()).float().mean()

            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("Accuracy/train", accuracy.item(), step)

            for tag, value in model.named_parameters():
                tag = tag.replace(".", "/")
                writer.add_histogram(tag, value.data.cpu().numpy(), step)
                if value.grad is not None:
                    writer.add_histogram(
                        tag + "/grad", value.grad.data.cpu().numpy(), step
                    )
                    

            avg_dw = 0.0
            for i, m in enumerate(model.rsnn.children()):
                if hasattr(m, 'avg_dw'):
                    avg_dw += m.avg_dw
            writer.add_scalar("Debug/DW", avg_dw/(i+1), step)

            
        if do_plot and batch_idx % plot_interval == 0:
            ts = np.arange(0, seq_length)
            fig, axs = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True)
            axs = axs.reshape(-1)  # flatten
            for nrn in range(10):
                one_trace = model.voltages.detach().cpu().numpy()[:, 0, nrn]
                fig.sca(axs[nrn])
                fig.plot(ts, one_trace)
            fig.xlabel("Time [s]")
            fig.ylabel("Membrane Potential")

            writer.add_figure("Voltages/output", fig, step)

        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss


def save(path, epoch, model, is_best=False):
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "is_best": is_best,
        },
        path,
    )


def load(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()
    return model


def main(argv):

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    kwargs = {"num_workers": 1, "pin_memory": True} if args.device == "cuda" else {}
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
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root=".",
            train=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        **kwargs,
    )

    label = os.environ.get("SLURM_JOB_ID", str(uuid.uuid4()))
    if args.prefix:
        path = f"runs/mnist_stdp/{args.prefix}"
    else:
        path = f"runs/mnist_stdp/"

    os.makedirs(path, exist_ok=True)

    logger.set_path(path)

    os.chdir(path)

    input_features = 28 * 28

    optimizer = STDPOptimizer()

    model = LIFConvNet(
        input_features,
        args.seq_length,
        input_scale=args.input_scale,
        model=args.method,
        only_first_spike=args.only_first_spike,
        optimizer=optimizer
    ).to(device)

    # No gradient required for STDP training
    model.no_grad()

    training_losses = []
    mean_losses = []
    test_losses = []
    accuracies = []

    for epoch in range(args.epochs):
        training_loss, mean_loss = train(
            model,
            optimizer,
            device,
            train_loader,
            epoch,
            clip_grad=args.clip_grad,
            grad_clip_value=args.grad_clip_value,
            epochs=args.epochs,
            log_interval=args.log_interval,
            do_plot=args.do_plot,
            plot_interval=args.plot_interval,
            seq_length=args.seq_length,
            writer=logger.writer(),
        )
        # test_loss, accuracy = test(
        #     model, device, test_loader, epoch, method=args.method, writer=writer
        # )

        training_losses += training_loss
        mean_losses.append(mean_loss)
        # accuracies.append(accuracy)

        # max_accuracy = np.max(np.array(accuracies))

        if (epoch % args.model_save_interval == 0) and args.save_model:
            model_path = f"mnist-{epoch}.pt"
            save(
                model_path,
                model=model,
                epoch=epoch,
                is_best=False,
            )

    np.save("training_losses.npy", np.array(training_losses))
    np.save("mean_losses.npy", np.array(mean_losses))
    np.save("accuracies.npy", np.array(accuracies))
    model_path = "mnist-final.pt"
    save(
        model_path,
        epoch=epoch,
        model=model,
        is_best=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        "MNIST digit recognition with convolutional SNN. Requires Tensorboard, Matplotlib, and Torchvision"
    )
    parser.add_argument(
        "--only-first-spike",
        type=bool,
        default=False,
        help="Only one spike per input (latency coding).",
    )
    parser.add_argument(
        "--save-grads",
        type=bool,
        default=False,
        help="Save gradients of backward pass.",
    )
    parser.add_argument(
        "--grad-save-interval",
        type=int,
        default=10,
        help="Interval for gradient saving of backward pass.",
    )
    parser.add_argument(
        "--refrac", type=bool, default=False, help="Use refractory time."
    )
    parser.add_argument(
        "--plot-interval", type=int, default=10, help="Interval for plotting."
    )
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help="Scaling factor for input current.",
    )
    parser.add_argument(
        "--find-learning-rate",
        type=bool,
        default=False,
        help="Use learning rate finder to find learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use by pytorch.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training episodes to do."
    )
    parser.add_argument(
        "--seq-length", type=int, default=200, help="Number of timesteps to do."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of examples in one minibatch.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="super",
        choices=["super", "tanh", "circ", "logistic", "circ_dist"],
        help="Method to use for training.",
    )
    parser.add_argument(
        "--prefix", type=str, default="", help="Prefix to use for saving the results"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="stdp",
        choices=["stdp", "rstdp"],
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--clip-grad",
        type=bool,
        default=False,
        help="Clip gradient during backpropagation",
    )
    parser.add_argument(
        "--grad-clip-value", type=float, default=1.0, help="Gradient to clip at."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-3, help="Learning rate to use."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="In which intervals to display learning progress.",
    )
    parser.add_argument(
        "--model-save-interval",
        type=int,
        default=50,
        help="Save model every so many epochs.",
    )
    parser.add_argument(
        "--save-model", type=bool, default=True, help="Save the model after training."
    )
    parser.add_argument("--big-net", type=bool, default=False, help="Use bigger net...")
    parser.add_argument(
        "--only-output", type=bool, default=False, help="Train only the last layer..."
    )
    parser.add_argument(
        "--do-plot", type=bool, default=False, help="Do intermediate plots"
    )
    parser.add_argument(
        "--random-seed", type=int, default=1234, help="Random seed to use"
    )
    args = parser.parse_args()
    main(args)
