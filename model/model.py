import torch
from torch import nn
import numpy as np
import torchlibrosa


# Encoder部分
# Encoder Block参数:
# inChannel = [2, 16, 32]
# outChannel = [16, 32, 64]
# kernelSize = (3, 5)
# padding = (1, 2)
# stride = [(1, 1), (1, 2), (1, 2)]
class ConvBlock(nn.Module):
    def __init__(self, inChannel, outChannel, kernelSize, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=inChannel, out_channels=outChannel,
                              kernel_size=kernelSize, stride=stride, bias=False
                              , padding=padding)
        self.BN = nn.BatchNorm2d(outChannel)
        self.pRelu = nn.PReLU(outChannel)

    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.pRelu(x)

        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        inChannel = [2, 16, 32]
        outChannel = [16, 32, 64]
        kernelSize = (3, 5)
        padding = (1, 2)
        stride = [(1, 1), (1, 2), (1, 2)]

        self.encoderBlock1 = ConvBlock(inChannel[0], outChannel[0], kernelSize, stride[0], padding)
        self.encoderBlock2 = ConvBlock(inChannel[1], outChannel[1], kernelSize, stride[1], padding)
        self.encoderBlock3 = ConvBlock(inChannel[2], outChannel[2], kernelSize, stride[2], padding)

    def forward(self, x):
        x = self.encoderBlock1(x)
        x = self.encoderBlock2(x)
        x = self.encoderBlock3(x)

        return x


# Decoder
# Decoder 参数：
# inChannel = [64, 32, 16]
# outChannel = [32, 16, 2]
# stride = [(1, 2), (1, 2), (1, 1)]
# outputPadding = [(0, 1), (0, 1), (0, 0)]
class GatedBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride, padding, outputPadding):
        super(GatedBlock, self).__init__()
        self.deConv = nn.ConvTranspose2d(in_channels=inChannel, out_channels=outChannel,
                                         kernel_size=(3, 5), stride=stride, padding=padding,
                                         output_padding=outputPadding)
        self.convBlock1 = ConvBlock(inChannel=2 * outChannel, outChannel=outChannel,
                                    kernelSize=(1, 1), stride=(1, 1), padding=(0, 0))
        self.convBlock2 = ConvBlock(inChannel=2 * outChannel, outChannel=outChannel,
                                    kernelSize=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x, residual):
        x = self.deConv(x)
        residual *= self.convBlock1(torch.cat((x, residual), dim=1))
        x += self.convBlock2(torch.cat((residual, x), dim=1))

        return x


# RA block
class ResidualBlock(nn.Module):
    def __init__(self, inChannel):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(inChannel, inChannel, (5, 7), (1, 1), (2, 3))
        self.conv2 = ConvBlock(inChannel, inChannel, (5, 7), (1, 1), (2, 3))

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual

        return x


class SelfAttention(nn.Module):
    def __init__(self, inChannel, numTokens):
        super(SelfAttention, self).__init__()
        self.convBlock1 = ConvBlock(inChannel, inChannel // 2, (1, 1), (1, 1), (0, 0))
        self.convBlock2 = ConvBlock(inChannel // 2, inChannel, (1, 1), (1, 1), (0, 0))
        self.qkvConvert = nn.Linear(numTokens, 3 * numTokens, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, d2, d3 = x.shape
        qkv = self.convBlock1(x)
        qkv = qkv.reshape(b, c // 2 * d2, d3)

        # 自注意力实现
        qkv = self.qkvConvert(qkv).reshape(b, c // 2 * d2, d3 * 3)
        q, k, v = qkv[:, :, :d3], qkv[:, :, d3: 2 * d3], qkv[:, :, 2 * d3:]
        del qkv

        att = (q @ k.transpose(-2, -1)) / np.sqrt(q.shape[-2])
        del q, k
        att = (att @ v).reshape(b, c // 2, d2, d3)
        x += self.convBlock2(att)

        return x


class RA_block(nn.Module):
    def __init__(self, inChannel, freTokens, timeTokens):
        super(RA_block, self).__init__()
        self.residualBlock1 = ResidualBlock(inChannel)
        self.residualBlock2 = ResidualBlock(inChannel)
        self.freAtt = SelfAttention(inChannel, freTokens)
        self.timeAtt = SelfAttention(inChannel, timeTokens)
        self.conv = ConvBlock(3 * inChannel, inChannel, (1, 1), (1, 1), 0)

    def forward(self, x):
        # two residual block
        x = self.residualBlock1(x)
        x = self.residualBlock2(x)

        # self-attention
        x = torch.cat((x, self.freAtt(x), self.timeAtt(x.transpose(-2, -1)).transpose(-2, -1)), dim=1)
        x = self.conv(x)

        return x


# interaction block
class InteractionBlock(nn.Module):
    def __init__(self, inChannel):
        super(InteractionBlock, self).__init__()
        self.conv = nn.Conv2d(inChannel * 2, inChannel, (1, 1), bias=False)
        self.BN = nn.BatchNorm2d(inChannel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = self.conv(torch.cat((x1, x2), dim=1))
        x = self.BN(x)
        x = self.sigmoid(x)
        x2 += (x * x1)

        return x2


# SN-Net:
class SN_net(nn.Module):
    def __init__(self, freBins, timeBins):
        super(SN_net, self).__init__()
        freBins = freBins // 2 // 4 + 1
        # STFT get real and image
        self.STFT = torchlibrosa.STFT(n_fft=320, hop_length=int(.01 * 16000),
                                      win_length=int(.02 * 16000), freeze_parameters=True)

        # Encoder: 3 encoder block
        inChannel = [2, 16, 32]
        outChannel = [16, 32, 64]
        kernelSize = (3, 5)
        padding = (1, 2)
        stride = [(1, 1), (1, 2), (1, 2)]

        self.encoderBlock1_S = ConvBlock(inChannel[0], outChannel[0], kernelSize, stride[0], padding)
        self.encoderBlock2_S = ConvBlock(inChannel[1], outChannel[1], kernelSize, stride[1], padding)
        self.encoderBlock3_S = ConvBlock(inChannel[2], outChannel[2], kernelSize, stride[2], padding)

        self.encoderBlock1_N = ConvBlock(inChannel[0], outChannel[0], kernelSize, stride[0], padding)
        self.encoderBlock2_N = ConvBlock(inChannel[1], outChannel[1], kernelSize, stride[1], padding)
        self.encoderBlock3_N = ConvBlock(inChannel[2], outChannel[2], kernelSize, stride[2], padding)

        # 4 RA block
        self.RA_block1_S = RA_block(outChannel[2], freBins, timeBins)
        self.RA_block2_S = RA_block(outChannel[2], freBins, timeBins)
        self.RA_block3_S = RA_block(outChannel[2], freBins, timeBins)
        self.RA_block4_S = RA_block(outChannel[2], freBins, timeBins)

        self.RA_block1_N = RA_block(outChannel[2], freBins, timeBins)
        self.RA_block2_N = RA_block(outChannel[2], freBins, timeBins)
        self.RA_block3_N = RA_block(outChannel[2], freBins, timeBins)
        self.RA_block4_N = RA_block(outChannel[2], freBins, timeBins)

        # 4 interaction block
        self.interactionBlock1 = InteractionBlock(outChannel[2])
        self.interactionBlock2 = InteractionBlock(outChannel[2])
        self.interactionBlock3 = InteractionBlock(outChannel[2])
        self.interactionBlock4 = InteractionBlock(outChannel[2])
        self.interactionBlock5 = InteractionBlock(outChannel[2])
        self.interactionBlock6 = InteractionBlock(outChannel[2])
        self.interactionBlock7 = InteractionBlock(outChannel[2])
        self.interactionBlock8 = InteractionBlock(outChannel[2])

        # Decoder: 3 gated block, 1 conv
        self.gatedBlock1_S = GatedBlock(outChannel[2], outChannel[1], (1, 2), (1, 2), (0, 0))
        self.gatedBlock2_S = GatedBlock(outChannel[1], outChannel[0], (1, 2), (1, 2), (0, 0))
        self.gatedBlock3_S = GatedBlock(outChannel[0], 2, (1, 1), (1, 2), (0, 0))
        self.conv_S = ConvBlock(2, 2, (1, 1), (1, 1), 0)

        self.gatedBlock1_N = GatedBlock(outChannel[2], outChannel[1], (1, 2), (1, 2), (0, 0))
        self.gatedBlock2_N = GatedBlock(outChannel[1], outChannel[0], (1, 2), (1, 2), (0, 0))
        self.gatedBlock3_N = GatedBlock(outChannel[0], 2, (1, 1), (1, 2), (0, 0))
        self.conv_N = ConvBlock(2, 2, (1, 1), (1, 1), 0)

        # ISTFT
        self.ISTFT = torchlibrosa.ISTFT(n_fft=320, hop_length=int(.01 * 16000),
                                      win_length=int(.02 * 16000), freeze_parameters=True)


    def forward(self, x):
        # STFT
        real, imag = self.STFT(x)
        x = torch.cat((real, imag), dim=1)
        del real, imag

        # speech branch encoder:
        speechResidua1 = self.encoderBlock1_S(x)
        speechResidua2 = self.encoderBlock2_S(speechResidua1)
        speechStem = self.encoderBlock3_S(speechResidua2)

        # noise branch encoder:
        noiseResidual1 = self.encoderBlock1_N(x)
        noiseResidual2 = self.encoderBlock2_N(noiseResidual1)
        noiseStem = self.encoderBlock3_N(noiseResidual2)

        # RA block and interaction
        speechStem = self.RA_block1_S(speechStem)
        noiseStem = self.RA_block1_N(noiseStem)
        temp = self.interactionBlock1(speechStem, noiseStem)
        speechStem = self.interactionBlock2(noiseStem, speechStem)
        noiseStem = temp

        speechStem = self.RA_block2_S(speechStem)
        noiseStem = self.RA_block2_N(noiseStem)
        temp = self.interactionBlock3(speechStem, noiseStem)
        speechStem = self.interactionBlock4(noiseStem, speechStem)
        noiseStem = temp

        speechStem = self.RA_block3_S(speechStem)
        noiseStem = self.RA_block3_N(noiseStem)
        temp = self.interactionBlock5(speechStem, noiseStem)
        speechStem = self.interactionBlock6(noiseStem, speechStem)
        noiseStem = temp

        speechStem = self.RA_block4_S(speechStem)
        noiseStem = self.RA_block4_N(noiseStem)
        temp = self.interactionBlock7(speechStem, noiseStem)
        speechStem = self.interactionBlock8(noiseStem, speechStem)
        noiseStem = temp
        del temp

        # Decoder:
        # speech decoder:
        speechStem = self.gatedBlock1_S(speechStem, speechResidua2)
        del speechResidua2
        speechStem = self.gatedBlock2_S(speechStem, speechResidua1)
        del speechResidua1
        speechStem = self.gatedBlock3_S(speechStem, x)
        speechStem = self.conv_S(speechStem)

        # noise decoder:
        noiseStem = self.gatedBlock1_N(noiseStem, noiseResidual2)
        del noiseResidual2
        noiseStem = self.gatedBlock2_N(noiseStem, noiseResidual1)
        del noiseResidual1
        noiseStem = self.gatedBlock3_N(noiseStem, x)
        del x
        noiseStem = self.conv_N(noiseStem)

        # ISTFT
        speech = self.ISTFT(speechStem[:, 0, :, :].unsqueeze(1), speechStem[:, 1, :, :].unsqueeze(1), None)
        noise = self.ISTFT(noiseStem[:, 0, :, :].unsqueeze(1), noiseStem[:, 1, :, :].unsqueeze(1), None)

        return {"speech spectrum": speechStem, "noise spectrum": noiseStem,
                "speech": speech, "noise": noise}


if __name__ == '__main__':
    # Encoder test
    # x = torch.ones(1, 2, 128, 320)  # (batch, channel, time, frequency)
    # encoder = Encoder()
    # y = encoder(x)
    # print(y.shape)

    # Decoder test
    # Decoder 参数：
    # inChannel = [64, 32, 16]
    # outChannel = [32, 16, 2]
    # stride = [(1, 2), (1, 2), (1, 1)]
    # outputPadding = [(0, 1), (0, 1), (0, 0)]
    # x = torch.ones(1, 64, 128, 80)
    # gatedBlock1 = GatedBlock(inChannel=inChannel[0], outChannel=outChannel[0],
    #                          stride=stride[0], padding=(1, 2), outputPadding=outputPadding[0])
    # gatedBlock2 = GatedBlock(inChannel[1], outChannel[1], stride=stride[1],
    #                          padding=(1, 2), outputPadding=outputPadding[1])
    # gatedBlock3 = GatedBlock(inChannel[2], outChannel[2], stride=stride[2],
    #                          padding=(1, 2), outputPadding=outputPadding[2])
    #
    # x = gatedBlock1(x)
    # x = gatedBlock2(x)
    # x = gatedBlock3(x)
    #
    # print(x.shape)

    # RA block test
    # x = torch.ones(1, 64, 128, 80)
    # att = RA_block(64, 80, 128)
    # y = att(x)
    # print(y.shape)

    # interaction block test
    # x1, x2 = torch.ones(1, 64, 128, 80), torch.ones(1, 64, 128, 80)
    # interactionBlock = InteractionBlock(64)
    # y = interactionBlock(x1, x2)
    # print(y.shape)

    # SN-Net test:
    snNet = SN_net(320, 201)
    x = torch.ones(8, 2 * 16000)  # (batch, channel, time, frequency)
    y = snNet(x)
    print(y["speech"].shape)
    print(y["noise"].shape)
    print(y["speech spectrum"].shape)
    print(y["noise spectrum"].shape)

