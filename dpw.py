import torch
import torch.nn as nn
import torch
# from torchsummary import summary



class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

if __name__=="__main__":
    # conv = DEPTHWISECONV(16, 64)
    # print(conv.depth_conv)
    # print(summary(conv,(16,1)))
    conv = depthwise_separable_conv(nin=4, nout=64, kernel_size = 5, padding = 0, bias=False)
    inp = torch.randn(32,4,20,20)
    out = conv(inp)
    print(out.shape)