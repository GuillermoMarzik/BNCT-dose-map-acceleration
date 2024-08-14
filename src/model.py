import torch
from torch import nn
from torch.nn import functional as F

class EncoderLayer(nn.Module):
    """ Encoder layer model """
    def __init__(self, in_channels, out_channels, stride=2, kernel_size=4, batchnorm=True):
        super().__init__()

        # Conv
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        # Weight init
        nn.init.normal_(self.conv.weight, std=0.02)
        # Batch normalization if applicable
        self.bn = nn.BatchNorm3d(out_channels) if batchnorm else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu_(x, negative_slope=0.2)
        return x


class DecoderLayer(nn.Module):
    """ Decoder layer model """
    def __init__(self, deconv_in_channels, deconv_out_channels, stride=2, kernel_size=4, dropout=False):
        super().__init__()

        # Transpose conv
        self.deconv = nn.ConvTranspose3d(in_channels=deconv_in_channels, out_channels=deconv_out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        # Weight init
        nn.init.normal_(self.deconv.weight, std=0.02)
        # Batch normalization
        self.bn = nn.BatchNorm3d(deconv_out_channels)
        # Dropout if applicable
        self.drop = nn.Dropout(p=0.5) if dropout else nn.Identity()

    def forward(self, x, skip):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.drop(x)
        if skip != None:
            x = torch.cat((x, skip), dim=1)
        x = F.leaky_relu_(x, negative_slope=0.2)
        return x

class SameDimLayerEnc(nn.Module):
    """ Layer that keeps dimensions constant"""
    def __init__(self, num_channels, num_layers, stride = 1, kernel_size = 3):
        # 3d conv
        self.conv = []
        self.bn = []
        self.num_l = num_layers
        for i in range(num_layers):
            self.conv.append(nn.Conv3d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, stride=stride, padding=1))
            # Weight init
            nn.init.normal_(self.conv[i].weight, std=0.02)
            # Batch normalization
            self.bn.append(nn.BatchNorm3d(num_channels))

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.conv[i](x)
            x = self.bn[i](x)
            #x = F.relu(x)
            x = F.leaky_relu_(x, negative_slope=0.2)
        return x

class EncConstDimBlock(nn.Module):
    def __init__(self, channels, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(
            channels, channels, stride=1, kernel_size=3)
            for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EncConstDimBlock_skip(nn.Module):
    def __init__(self, channels, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(
            channels, channels, stride=1, kernel_size=3)
            for i in range(num_layers)])

    def forward(self, x):
        x_out = [x]
        for layer in self.layers:
            x_out.append(layer(x_out[-1]))
        return x_out

class DecConstDimBlock(nn.Module):
    def __init__(self, channels,num_layers):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(
            channels, channels, stride=1, kernel_size=3)
            for i in range(num_layers)])   

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, None)
        return x

class DecConstDimBlock_skip(nn.Module):
    def __init__(self, channels,num_layers):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(
            channels, channels//2, stride=1, kernel_size=3)
            for i in range(num_layers)])   

    def forward(self, x, skip):
        for i in range(len(self.layers)):
            x = self.layers[i](x, skip[-i-1])
        return x

class SameDimLayerDec(nn.Module):
    """ Layer that keeps dimensions constant"""
    def __init__(self, num_channels, num_layers, stride = 1, kernel_size = 3):
        # 3d conv
        self.deconv = []
        self.bn = []
        self.num_l = num_layers
        for i in range(num_layers):
            self.deconv.append(nn.ConvTranspose3d(in_channels=num_channels, out_channels=num_channles, kernel_size=kernel_size, stride=stride, padding=1))
            # Weight init
            nn.init.normal_(self.deconv[i].weight, std=0.02)
            # Batch normalization
            self.bn.append(nn.BatchNorm3d(num_channels))

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.deconv[i](x)
            x = self.bn[i](x)
            #x = F.relu(x)
            x = F.leaky_relu_(x, negative_slope=0.2)
        return x


class Unet_skip(nn.Module):
    def __init__(self, num_layers, input_shape, output_shape):
        super().__init__()

        # Encoder blocks
        self.e1 = EncoderLayer(in_channels=input_shape, out_channels=64, batchnorm=False)
        self.e2 = EncConstDimBlock_skip(channels=64, num_layers=num_layers) 
        self.e3 = EncoderLayer(in_channels=64,  out_channels=128)
        self.e4 = EncConstDimBlock_skip(channels=128, num_layers=num_layers)
        # Bottleneck
        self.bottleneck = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
        nn.init.normal_(self.bottleneck.weight, std=0.02)
        # Decoder blocks
        self.d1 = DecoderLayer(deconv_in_channels=128, deconv_out_channels=128)
        self.d2 = DecConstDimBlock_skip(channels=256, num_layers=num_layers)
        self.d3 = DecoderLayer(deconv_in_channels=256, deconv_out_channels=64)
        self.d4 = DecConstDimBlock_skip(channels=128, num_layers=num_layers)
        # Output
        self.out = nn.ConvTranspose3d(in_channels=128, out_channels=output_shape, kernel_size=4, stride=2, padding=1)
        # Weight init
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x):
        # Encoder
        e1_out = self.e1(x)
        e2_out = self.e2(e1_out)
        e3_out = self.e3(e2_out[-1])
        e4_out = self.e4(e3_out)
        # Bottleneck
        b = F.leaky_relu(self.bottleneck(e4_out[-1]), negative_slope=0.1)
        # Decoder
        d1_out = self.d1(b, e4_out[0])
        d2_out = self.d2(d1_out, e4_out)
        d3_out = self.d3(d2_out, e2_out[0])
        d4_out = self.d4(d3_out, e2_out)
        # Output
        out = F.leaky_relu(self.out(d4_out), negative_slope=0.1)
        return out
