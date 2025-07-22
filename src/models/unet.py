"""
"U-Net: Convolutional Networks for Biomedical Image Segmentation"
by Ronneberger et al., 2015.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net.
    
    Performs two consecutive 3x3 convolutions, each followed by
    batch normalization and ReLU activation.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        mid_channels: Number of channels for the intermediate layer.
            If None, uses out_channels.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        mid_channels: Optional[int] = None
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the double convolution block.
        
        x: Input tensor of shape (batch_size, in_channels, height, width)
        Output tensor of shape (batch_size, out_channels, height, width)
        """
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block for U-Net encoder.
    
    Performs max pooling followed by double convolution.
    
    in_channels: Number of input channels.
    out_channels: Number of output channels.
    """
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the downsampling block.
        
        x: Input tensor of shape (batch_size, in_channels, height, width)
        Output tensor with shape (batch_size, out_channels, height//2, width//2)
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block for U-Net decoder.
    
    Performs upsampling (transpose convolution or bilinear interpolation)
    followed by concatenation with skip connection and double convolution.
    
    in_channels: Number of input channels.
    out_channels: Number of output channels.
    bilinear: If True, use bilinear interpolation for upsampling.
        Otherwise, use transpose convolution.
    """
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
            
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass through the upsampling block.
        
        x1: Input tensor from the previous decoder layer.
        x2: Skip connection tensor from the corresponding encoder layer.
        Output tensor after upsampling, concatenation, and convolution.
        """
        x1 = self.up(x1)
        
        # Handle cases where input sizes don't match perfectly
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution for final segmentation map.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (classes).
    """
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the output convolution.
        
        x: Input tensor.
        Output tensor with shape (batch_size, out_channels, height, width).
        """
        return self.conv(x)


class UNet(nn.Module):
    """U-Net model for image segmentation.
    
    The U-Net architecture consists of an encoder (contracting path) and
    decoder (expanding path) with skip connections between corresponding levels.
    
    n_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB).
    n_classes: Number of output classes for segmentation.
    bilinear: If True, use bilinear upsampling. Otherwise, use transpose conv.
    base_features: Number of features in the first layer. Doubled at each level.
        
    Example:
        >>> model = UNet(n_channels=1, n_classes=1)  # Binary segmentation
        >>> input_tensor = torch.randn(4, 1, 256, 256)
        >>> output = model(input_tensor)
        >>> print(output.shape)  # torch.Size([4, 1, 256, 256])
    """
    
    def __init__(
        self, 
        n_channels: int = 1, 
        n_classes: int = 1, 
        bilinear: bool = True,
        base_features: int = 64
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Calculate feature sizes for each layer
        features = base_features
        
        # Encoder 
        self.inc = DoubleConv(n_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor)
        
        # Decoder 
        self.up1 = Up(features * 16, features * 8 // factor, bilinear)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear)
        self.up4 = Up(features * 2, features, bilinear)
        self.outc = OutConv(features, n_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the U-Net.
        
        x: Input tensor of shape (batch_size, n_channels, height, width).
        Output tensor of shape (batch_size, n_classes, height, width).
        For binary segmentation with n_classes=1, apply sigmoid activation
        to get probabilities.
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def get_num_params(self) -> int:
        """Get the total number of trainable parameters.
        
        Total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 