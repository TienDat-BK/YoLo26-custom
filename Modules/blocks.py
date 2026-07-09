from torch import autograd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

        # đã CÀI ATTENTION CHO C3K2 Block
        self.m = nn.ModuleList(
            nn.Sequential(
                BottleNeck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else BottleNeck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

class SPPF(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, k: int = 5, n: int = 3, shortcut: bool = True):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            in_channel (int): Input channels.
            out_channel (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.
        """
        super().__init__()

        c_ = in_channel // 2  # hidden channels
        self.cv1 = Conv(in_channel, c_, 1, 1)
        self.cv2 = Conv(c_ * (n + 1), out_channel, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and in_channel == out_channel
    
    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        res = self.cv2(torch.cat(y, 1))
        return x + res if self.add else res

class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_dim = int(self.head_dim * attn_ratio)
        self.qk_dim_total = self.qk_dim * num_heads
        self.qkv_dim_out = dim + self.qk_dim_total * 2
        self.scale = self.qk_dim ** -0.5

        self.qkv = Conv(in_channel=dim, out_channel=self.qkv_dim_out, k=1,s=1,act=False)
        self.proj = Conv(in_channel=dim, out_channel=dim, k = 1, act=False)
        self.position_embedder = Conv(in_channel=dim, out_channel=dim, k = 3, s=1, g=dim, act=False)
    
    def forward(self, x : torch.Tensor):
        # x : (B, C, H, W)
        B, C, H, W = x.shape

        x = self.qkv(x)
        # x : (B, qkv_dim_out, H*W)

        x = x.view(B, self.num_heads, self.qkv_dim_out // self.num_heads, -1)
        # x : (B, num_heads, qkv_dim_out // num_heads, H*W)

        Q, K, V = x.split([self.qk_dim, self.qk_dim, self.head_dim], dim= -2)
        
        x = (V @ F.softmax(((Q.transpose(-1,-2) @ K) * self.scale), dim=-1).transpose(-1, -2)) 
        # x : (B, num_heads, head_dim, H*W)
        
        x = x.view(B, self.dim, H, W) + self.position_embedder(V.reshape(B, self.dim, H, W))
        # x : (B, dim, H, W)

        return self.proj(x)

class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))        

        

