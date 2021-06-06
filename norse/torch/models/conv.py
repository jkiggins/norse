import torch

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFCell
from norse.torch.module.stdp import STDPModule

from norse.torch.functional.stdp import stdp_step_linear, stdp_step_conv2d, STDPState, STDPParameters


class ConvNet(torch.nn.Module):
    """
    A convolutional network with LIF dynamics

    Arguments:
        num_channels (int): Number of input channels
        feature_size (int): Number of input features
        method (str): Threshold method
    """

    def __init__(
        self, num_channels=1, feature_size=28, method="super", dtype=torch.float
    ):
        super(ConvNet, self).__init__()
        self.features = int(((feature_size - 4) / 2 - 4) / 2)
        self.conv1 = torch.nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(self.features * self.features * 50, 500)
        self.out = LILinearCell(500, 10)
        self.lif0 = LIFCell(
            p=LIFParameters(method=method, alpha=100.0),
        )
        self.lif1 = LIFCell(
            p=LIFParameters(method=method, alpha=100.0),
        )
        self.lif2 = LIFCell(p=LIFParameters(method=method, alpha=100.0))
        self.dtype = dtype

        
    def forward(self, x, stdp=False):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        s0, s1, s2, so = None, None, None, None

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=self.dtype
        )

        for ts in range(seq_length):
            z = x[ts, :]
            z_pre = z
            
            z = self.conv1(z)
            z, s0 = self.lif0(z, s0)
            if stdp: self.stdp_step(self.lif0, z_pre, z)
            
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z_pre = z
            z = 10 * self.conv2(z)
            z, s1 = self.lif1(z, s1)
            if stdp: self.stdp_step(self.lif1, z_pre, z)
                           
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = z.view(-1, self.features ** 2 * 50)
            z_pre = z
            z = self.fc1(z)
            z, s2 = self.lif2(z, s2)
            if stdp: self.stdp_step(self.lif2, z_pre, z)
            
            v, so = self.out(torch.nn.functional.relu(z), so)
            voltages[ts, :, :] = v
        return voltages


class ConvNet4(torch.nn.Module):
    """
    A convolutional network with LIF dynamics

    Arguments:
        num_channels (int): Number of input channels
        feature_size (int): Number of input features
        method (str): Threshold method
    """

    def __init__(
        self, num_channels=1, feature_size=28, method="super", dtype=torch.float
    ):
        super(ConvNet4, self).__init__()
        self.features = int(((feature_size - 4) / 2 - 4) / 2)

        self.conv1 = torch.nn.Conv2d(num_channels, 32, 5, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, 1)
        self.fc1 = torch.nn.Linear(self.features * self.features * 64, 1024)
        self.lif0 = LIFCell(
            p=LIFParameters(method=method, alpha=100.0, v_th=torch.as_tensor(0.7)),
        )
        self.lif1 = LIFCell(
            p=LIFParameters(method=method, alpha=100.0, v_th=torch.as_tensor(0.7)),
        )
        self.lif2 = LIFCell(p=LIFParameters(method=method, alpha=100.0))
        self.out = LILinearCell(1024, 10)
        self.dtype = dtype


    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        s0, s1, s2, so = None, None, None, None

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=self.dtype
        )

        for ts in range(seq_length):
            z = self.conv1(x[ts, :])
            z, s0 = self.lif0(z, s0)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = 10 * self.conv2(z)
            z, s1 = self.lif1(z, s1)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = z.view(-1, self.features ** 2 * 64)
            z = self.fc1(z)
            z, s2 = self.lif2(z, s2)
            v, so = self.out(torch.nn.functional.relu(z), so)
            voltages[ts, :, :] = v
        return voltages


class ConvNetStdp(torch.nn.Module, STDPModule):
    def __init__(
        self, num_channels=1, feature_size=28, method="super", dtype=torch.float
    ):
        torch.nn.Module.__init__(self)
        STDPModule.__init__(self)

        self.features = int(((feature_size - 4) / 2 - 4) / 2)

        self.conv1 = torch.nn.Conv2d(num_channels, 32, 5, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, 1)
        self.fc1 = torch.nn.Linear(self.features * self.features * 64, 1024)
        self.lif0 = LIFCell(
            p=LIFParameters(method=method, alpha=100.0),
        )
        self.lif1 = LIFCell(
            p=LIFParameters(method=method, alpha=100.0),
        )
        self.lif2 = LIFCell(p=LIFParameters(method=method, alpha=100.0))
        self.out = LILinearCell(1024, 10)
        self.dtype = dtype

        
    def forward(self, x, stdp=False):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        s0 = None
        s1 = None
        s2 = None
        so = None

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=self.dtype
        )


        for ts in range(seq_length):
            z = x[ts, :]
            z_pre = z
            
            z = self.conv1(z)
            z, s0 = self.lif0(z, s0)
            if stdp: self.stdp_step(self.conv1, z_pre, z)
            
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z_pre = z
            z = 10 * self.conv2(z)
            z, s1 = self.lif1(z, s1)
            if stdp: self.stdp_step(self.conv2, z_pre, z)
                           
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = z.view(-1, self.features ** 2 * 64)
            z_pre = z
            z = self.fc1(z)
            z, s2 = self.lif2(z, s2)
            if stdp: self.stdp_step(self.fc1, z_pre, z)
            
            v, so = self.out(torch.nn.functional.relu(z), so)
            voltages[ts, :, :] = v

        return voltages


class ConvNetStdp(torch.nn.Module, STDPModule):
    def __init__(
        self, num_channels=1, feature_size=28, method="super", dtype=torch.float
    ):
        torch.nn.Module.__init__(self)
        STDPModule.__init__(self)

        self.features = int(((feature_size - 4) / 2 - 4) / 2)

        self.conv1 = torch.nn.Conv2d(num_channels, 32, 5, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, 1)
        self.fc1 = torch.nn.Linear(self.features * self.features * 64, 1024)
        self.lif0 = LIFCell(
            p=LIFParameters(method=method, alpha=100.0),
        )
        self.lif1 = LIFCell(
            p=LIFParameters(method=method, alpha=100.0),
        )
        self.lif2 = LIFCell(p=LIFParameters(method=method, alpha=100.0))
        self.out = LILinearCell(1024, 10)
        self.dtype = dtype

        
    def forward(self, x, stdp=False):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        s0 = None
        s1 = None
        s2 = None
        so = None

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=self.dtype
        )


        for ts in range(seq_length):
            z = x[ts, :]
            z_pre = z
            
            z = self.conv1(z)
            z, s0 = self.lif0(z, s0)
            if stdp: self.add_stdp_step(self.conv1, z_pre, z)
            
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z_pre = z
            z = 10 * self.conv2(z)
            z, s1 = self.lif1(z, s1)
            if stdp: self.add_stdp_step(self.conv2, z_pre, z)
                           
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = z.view(-1, self.features ** 2 * 64)
            z_pre = z
            z = self.fc1(z)
            z, s2 = self.lif2(z, s2)
            if stdp: self.add_stdp_step(self.fc1, z_pre, z)
            
            v, so = self.out(torch.nn.functional.relu(z), so)
            voltages[ts, :, :] = v

            if stdp: self.stdp_step()

        return voltages
