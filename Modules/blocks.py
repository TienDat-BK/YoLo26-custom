from torch import autograd
import torch
import torch.nn as nn

def autopad(k, p=None, d=1) -> int | tuple[int, int]:  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p 

class Conv(nn.Module):
    def_act = nn.SiLU()

    def __init__(self, in_channel : int, out_channel : int , k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        """Initialize Conv layer with given parameters.
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """

        self.cv = nn.Conv2d(in_channel, out_channel, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = self.def_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.cv(x)))

class BottleNeck(nn.Module):
    def __init__(self, c1 : int, c2 : int, shortcut : bool = True, g : int = 1, k : tuple[int, int] = (3, 3), e = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.c1 = Conv(c1, c_, k=k[0], s = 1)
        self.c2 = Conv(c_, c2, k=k[1], s = 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.c2(self.c1(x)) if self.add else self.c2(self.c1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, in_channel: int, out_channel: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            in_channel (int): Input channels.
            out_channel (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(out_channel * e)
        self.cv1 = Conv(in_channel, 2 * self.c, k = 1, s = 1)
        self.cv2 = Conv((2 + n) * self.c, out_channel, k = 1, s = 1)
        self.m = nn.ModuleList(BottleNeck(self.c, self.c, shortcut, k = (3, 3), e = 1.0) for _ in range(n))
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend( m(y[-1]) for m in self.m )
        return self.cv2( torch.cat(y, 1) )

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(BottleNeck(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(BottleNeck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2(C2f):
    def __init__(self, in_channels : int, out_channel : int, n : int = 1, c3k : bool = True, e : float = 0.5, attn : bool = False, g : int = 1, shortcut : bool = True):
        
        """Initialize C3k2 module.

        Args:
            in_channels (int): Input channels.
            out_channel (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(in_channels, out_channel, n = n, g = g, e = e, shortcut = shortcut)

        # CHƯA CÀI ATTENTION CHO C3K2 Block
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else BottleNeck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

a = C3k2(in_channels=2, out_channel= 4 , n = 2, g=1, shortcut=True)


        

