import torch
import torch.nn as nn


def GlobalMind(args):
    if args.GAS_mode == 'GRS-GRS':  # 行-行
        model = Model_GlobalMind(GlobalM_GRS, GlobalM_GRS,GlobalD_GRS, GlobalD_GRS,
                                             args.layernum, args.pixel_num, hidn_dim=256)
    elif args.GAS_mode == 'GRS-GCS':  # 行-列
        model = Model_GlobalMind(GlobalM_GRS, GlobalM_GCS,GlobalD_GRS, GlobalD_GCS,
                                             args.layernum, args.pixel_num, hidn_dim=256)
    elif args.GAS_mode == 'GCS-GRS':  # 列-行
        model = Model_GlobalMind(GlobalM_GCS, GlobalM_GRS,GlobalD_GCS, GlobalD_GRS,
                                             args.layernum, args.pixel_num, hidn_dim=256)
    elif args.GAS_mode == 'GCS-GCS':  # 列-列
        model = Model_GlobalMind(GlobalM_GCS, GlobalM_GCS,GlobalD_GCS, GlobalD_GCS,
                                             args.layernum, args.pixel_num,hidn_dim=256)
    else:
        raise ValueError("args.GAS_mode should be one of the {'GRS-GRS', 'GRS-GCS', 'GCS-GRS','GCS-GCS'}.")
    return model


class Model_GlobalMind(nn.Module):
    def __init__(self, spAtt1, spAtt2, intAtt1, intAtt2,layernum, pixel_num, hidn_dim=256):
        super(Model_GlobalMind, self).__init__()

        self.conv1 = FeatureBlock(layernum[0], layernum[1])
        self.conv2 = FeatureBlock(2 * layernum[1], layernum[2])
        self.conv3 = FeatureBlock(2 * layernum[2], layernum[3])
        self.Spatial_selfatt1 = self._make_spAtt_layer(spAtt1, in_dim=layernum[1], out_dim=layernum[1],
                                                   pixel_num=pixel_num, hidn_dim=hidn_dim)
        self.Spatial_selfatt2 = self._make_spAtt_layer(spAtt2, in_dim=layernum[2], out_dim=layernum[2],
                                                   pixel_num=pixel_num, hidn_dim=hidn_dim)

        self.Temporal_interatt1 = self._make_intAtt_layer(intAtt1, in_dim=layernum[2], out_dim=layernum[2], hidn_dim=256)
        self.Temporal_interatt2 = self._make_intAtt_layer(intAtt1, in_dim=layernum[4], out_dim=layernum[4], hidn_dim=256)
        self.Temporal_interatt3 = self._make_intAtt_layer(intAtt2, in_dim=layernum[6], out_dim=layernum[6], hidn_dim=256)

        in_dim = layernum[2] + layernum[4] + layernum[6]
        self.CD = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=(1,1), bias=False),
                                nn.ReLU(inplace=True),  # third layer
                                nn.Dropout(0.1),
                                nn.Conv2d(in_dim, in_dim//2, kernel_size=(1,1), bias=False),
                                nn.ReLU(inplace=True),  # third layer
                                nn.Dropout(0.1),
                                nn.Conv2d(in_dim//2, 2, kernel_size=(1,1), bias=True))
    def _make_spAtt_layer(self, block, in_dim, out_dim,pixel_num,hidn_dim):
        layers = []
        layers.append(block(in_dim, out_dim, pixel_num, hidn_dim))
        return nn.Sequential(*layers)
    def _make_intAtt_layer(self, block, in_dim, out_dim, hidn_dim):
        layers = nn.ModuleList()
        layers.append(block(in_dim, out_dim, hidn_dim))
        return layers[0]

    def cross_entropy(self, loss_fuc1, result, label, idx):
        # loss_fuc1: CrossEntropyLoss
        # result: [2, H,W]; label[N]
        num, H, B = result.shape
        result = result.reshape([num, -1])  # [2, H*W]
        result_dx = result[:, idx]   # [2, N]
        result_dx = result_dx.transpose(1, 0)   # [N, 2]
        l_ce = loss_fuc1(result_dx, label.squeeze())
        return l_ce

    def getFeature(self,x):
        f1 = self.conv1(x)  # B, C, H,W
        f1_ = self.Spatial_selfatt1(f1)
        f1_ = torch.cat([f1, f1_], dim=1)
        f2 = self.conv2(f1_)
        f2_ = self.Spatial_selfatt2(f2)
        f2_ = torch.cat([f2,f2_], dim=1)
        f3 = self.conv3(f2_)
        return f1,f2,f3
    def getDifference(self,f10, f20,f11, f21,f12, f22):
        f1 = self.Temporal_interatt1(f10, f20)
        f2 = self.Temporal_interatt2(f11, f21)
        f3 = self.Temporal_interatt3(f12, f22)
        f = torch.cat([f1, f2, f3], dim=1)  # B,C,H,W
        out = self.CD(f).squeeze()  # 2,H,W
        return out

    # to visiualize the feature map
    def forward(self, t1, t2, idx, label, loss_fuc1):
        f10, f11, f12 = self.getFeature(t1)
        f20, f21, f22 = self.getFeature(t2)
        self.feature1 = f10
        self.feature2 = f20
        output1 = self.getDifference(f10, f20,f11, f21,f12, f22)
        output2 = self.getDifference(f20, f10, f21, f11, f22, f12)
        if self.training:
            l_ce1 = self.cross_entropy(loss_fuc1, output1, label, idx)
            l_ce2 = self.cross_entropy(loss_fuc1, output2, label, idx)
            return (l_ce1+l_ce2)/2
        return output1, output2

# GlobalM: global spatial multi-head interactive self-attention (GlobalM) module
# Global Row Segmentation (GRS)
class GlobalM_GRS(nn.Module):
    def __init__(self, in_dim, out_dim, pixel_num,hidn_dim=256):
        super(GlobalM_GRS, self).__init__()
        self.norm1 = nn.LayerNorm(pixel_num, elementwise_affine=False)
        self.query_conv = nn.Conv2d(in_dim, in_dim, kernel_size=(1,1),stride=(1,1), bias=True, groups=in_dim)
        self.key_conv = nn.Conv2d(in_dim, in_dim, kernel_size=(1,1),stride=(1,1), bias=True, groups=in_dim)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=(1,1),stride=(1,1), bias=True, groups=in_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(0.1)
        self.inter = conv3x3(in_dim, in_dim)
        if in_dim != out_dim:
            self.downsample = conv1x1(in_planes=in_dim, out_planes=out_dim)
        else:
            self.downsample = None

        self.norm2 = nn.LayerNorm(pixel_num, elementwise_affine=False)
        self.mlp = nn.Sequential(
            conv1x1(in_planes=out_dim, out_planes=hidn_dim),
            nn.ReLU(),
            conv1x1(in_planes=hidn_dim, out_planes=out_dim)
        )
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        batchsize, C, height, width = x.size()
        # head_num = channel; seq_len=H, embd_dim=W
        x0 = self.norm1(x.view(batchsize, -1, width * height))   # batchsize, C, height*width
        q = self.query_conv(x0.view(batchsize, -1, height, width))  # B,C,H,W
        k = self.key_conv(x0.view(batchsize, -1, height, width))  # B,C,H,W
        v = self.value_conv(x0.view(batchsize, -1, height, width))  # B,C,H,W

        scale = width ** -0.5  # n_head = height, then head_dim = width
        attn = (q @ k.transpose(-2, -1)) * scale  # batchsize, C, H,H
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)  # [B,C, H,H]
        out = (attn @ v) # B,C,H,W
        out = self.inter(out)
        if self.downsample is not None:
            x = self.downsample(x)
        x = out + x

        x0 = self.norm2(x.reshape(batchsize, C, width*height))
        x0 = self.mlp(x0.view(batchsize, -1, height,width))
        x0 = self.proj_drop(x0)
        out = x0 + x
        return out
# GlobalM: global spatial multi-head interactive self-attention (GlobalM) module
# Global Column Segmentation (GCS)
class GlobalM_GCS(nn.Module):
    def __init__(self, in_dim, out_dim, pixel_num,hidn_dim=256):
        super(GlobalM_GCS, self).__init__()
        self.norm1 = nn.LayerNorm(pixel_num, elementwise_affine=False)
        self.query_conv = nn.Conv2d(in_dim, in_dim, kernel_size=(1,1),stride=(1,1), bias=True, groups=in_dim)
        self.key_conv = nn.Conv2d(in_dim, in_dim, kernel_size=(1,1),stride=(1,1), bias=True, groups=in_dim)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=(1,1),stride=(1,1), bias=True, groups=in_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(0.1)
        self.inter = conv3x3(in_dim, in_dim)
        if in_dim != out_dim:
            self.downsample = conv1x1(in_planes=in_dim, out_planes=out_dim)
        else:
            self.downsample = None

        self.norm2 = nn.LayerNorm(pixel_num, elementwise_affine=False)
        self.mlp = nn.Sequential(
            conv1x1(in_planes=out_dim, out_planes=hidn_dim),
            nn.ReLU(),
            conv1x1(in_planes=hidn_dim, out_planes=out_dim)
        )
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        # x: batchsize, C, height, width
        x = x.transpose(-2, -1) # batchsize, C, width, height
        batchsize, C, height, width = x.size()
        # head_num = channel; seq_len=H, embd_dim=W
        x0 = x.reshape(batchsize, -1, width * height)
        x0 = self.norm1(x0)   # batchsize, C, height*width
        q = self.query_conv(x0.view(batchsize, -1, height, width))  # B,C,H,W
        k = self.key_conv(x0.view(batchsize, -1, height, width))  # B,C,H,W
        v = self.value_conv(x0.view(batchsize, -1, height, width))  # B,C,H,W

        scale = width ** -0.5  # n_head = height, then head_dim = width
        attn = (q @ k.transpose(-2, -1)) * scale  # batchsize, C, H,H
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)  # [B,C, H,H]
        out = (attn @ v) # B,C,H,W
        out = self.inter(out)
        if self.downsample is not None:
            x = self.downsample(x)
        x = out + x

        x0 = self.norm2(x.reshape(batchsize, C, width*height))
        x0 = self.mlp(x0.view(batchsize, -1, height,width))
        x0 = self.proj_drop(x0)
        out = x0 + x
        out = out.transpose(-2, -1)  # batchsize, C, width, height
        return out
# GlobalD: global temporal interactive multi-head selfattention (GlobalD) module
# Global Row Segmentation (GRS)
class GlobalD_GRS(nn.Module):
    def __init__(self, in_dim,out_dim,hidn_dim):
        super(GlobalD_GRS, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = conv1x1(in_planes=in_dim, out_planes=in_dim)  # out_planes=in_dim//2
        self.key_conv = conv1x1(in_planes=in_dim, out_planes=in_dim)
        self.value_conv = conv1x1(in_planes=in_dim, out_planes=in_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.inter = conv3x3(in_dim, in_dim)
        self.mlp = nn.Sequential(
            conv1x1(in_planes=in_dim, out_planes=hidn_dim),
            nn.ReLU(),
            conv1x1(in_planes=hidn_dim, out_planes=out_dim)
        )

    def forward(self, x, y):
        batchsize, C, height, width = x.size()
        # head_num = channel; seq_len=H, embd_dim=W
        q = self.query_conv(x)  # B,C,H,W;
        k = self.key_conv(y)  # B,C,H,W
        diff = torch.abs(x - y)
        v = self.value_conv(diff)  # B,C,H,W
        scale = (width * height) ** -0.5  # feature dim: width * height; sequence length: channel
        attn = (q @ k.transpose(-2, -1)) * scale # B,C,H,H
        attn = self.softmax(attn)   # batchsize, C,C
        x = (attn @ v) # B,C,H,W
        x = self.inter(x)
        x = x + diff # batchsize, C, height, width
        x = self.mlp(x)
        return x
# GlobalD: global temporal interactive multi-head selfattention (GlobalD) module
# Global Column Segmentation (GCS)
class GlobalD_GCS(nn.Module):
    def __init__(self, in_dim,out_dim,hidn_dim):
        super(GlobalD_GCS, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = conv1x1(in_planes=in_dim, out_planes=in_dim)  # out_planes=in_dim//2
        self.key_conv = conv1x1(in_planes=in_dim, out_planes=in_dim)
        self.value_conv = conv1x1(in_planes=in_dim, out_planes=in_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.inter = conv3x3(in_dim, in_dim)
        self.mlp = nn.Sequential(
            conv1x1(in_planes=in_dim, out_planes=hidn_dim),
            nn.ReLU(),
            conv1x1(in_planes=hidn_dim, out_planes=out_dim)
        )

    def forward(self, x, y):
        # x : batchsize, C,  height, width
        x = x.transpose(-2, -1)  # batchsize, C, width, height
        y = y.transpose(-2, -1)  # batchsize, C, width, height
        batchsize, C, height, width = x.size()
        # head_num = channel; seq_len=H, embd_dim=W
        q = self.query_conv(x)  # B,C,H,W;
        k = self.key_conv(y)  # B,C,H,W
        diff = torch.abs(x - y)
        v = self.value_conv(diff)  # B,C,H,W
        scale = (width * height) ** -0.5  # feature dim: width * height; sequence length: channel
        attn = (q @ k.transpose(-2, -1)) * scale # B,C,H,H
        attn = self.softmax(attn)   # batchsize, C,C
        x = (attn @ v) # B,C,H,W
        x = self.inter(x)
        x = x + diff # batchsize, C, height, width
        x = self.mlp(x)
        x = x.transpose(-2, -1)  # batchsize, C, height, width
        return x



class FeatureBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FeatureBlock, self).__init__()
        self.conv11 = conv1x1(in_planes, out_planes)
        self.bn11 = nn.BatchNorm2d(out_planes)
        self.relu11 = nn.ReLU()
        self.conv12 = conv3x3(out_planes, out_planes//2)

        self.conv21 = conv3x3(in_planes, out_planes)
        self.bn21 = nn.BatchNorm2d(out_planes)
        self.relu21 = nn.ReLU()
        self.conv22 = conv3x3(out_planes, out_planes//2)

    def forward(self,x):
        f1 = self.relu11(self.bn11(self.conv11(x)))
        f1 = self.conv12(f1)
        f2 = self.relu21(self.bn21(self.conv21(x)))
        f2 = self.conv22(f2)
        x = torch.cat([f1, f2],dim=1)
        return x  # [B, out_planes, H,W]
def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,1), stride=(stride,stride),
                     padding=0, bias=False)
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=(stride,stride),
                     padding=1, bias=False)




