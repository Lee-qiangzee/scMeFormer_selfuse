import torch
import torch.nn as nn
import torch.nn.functional as F

class DNAModule(nn.Module):
    def __init__(self, n_channels=4, cnn_out=256, cnn_kernel=10, cnn_layers=3, maxpool_stride=20,
                 cnn_dropout=0.5, transformer_layers=8, nhead=8, trans_dropout=0.1):
        super().__init__()
        convs = []
        in_channels = n_channels
        for _ in range(cnn_layers):
            convs.append(nn.Conv1d(in_channels, cnn_out, cnn_kernel, padding=cnn_kernel//2))
            convs.append(nn.BatchNorm1d(cnn_out))
            convs.append(nn.ReLU())
            in_channels = cnn_out
        self.cnn = nn.Sequential(*convs)
        self.pool = nn.MaxPool1d(kernel_size=maxpool_stride, stride=maxpool_stride)
        self.cnn_dropout = nn.Dropout(cnn_dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=cnn_out, nhead=nhead, 
                                                   dim_feedforward=cnn_out*4, 
                                                   dropout=trans_dropout, 
                                                   activation='relu', 
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

    def forward(self, x):
        # x: (batch_size, segment_size, 2000, 4)
        batch_size, segment_size, seq_len, n_ch = x.shape
        x = x.reshape(-1, seq_len, n_ch)  # (batch_size*segment_size, 2000, 4)
        x = x.permute(0, 2, 1)  # (batch_size*segment_size, 4, 2000)
        x = self.cnn(x)  # (batch_size*segment_size, 256, 2000)
        x = self.pool(x)  # (batch_size*segment_size, 256, 100)
        x = self.cnn_dropout(x)
        x = x.permute(0, 2, 1)  # (batch_size*segment_size, 100, 256)
        x = self.transformer(x)  # (batch_size*segment_size, 100, 256)
        x = x.mean(dim=1)  # (batch_size*segment_size, 256)
        x = x.view(batch_size, segment_size, -1)  # (batch_size, segment_size, 256)
        return x

class CpGModuleWithWindow(nn.Module):
    def __init__(
        self,
        segment_size: int,
        n_cluster: int,
        window_size: int = 101,
        cnn_out: int = 256,
        cnn_kernel: int = 10,
        cnn_layers: int = 3,
        cnn_dropout: float = 0.5,
        trans_layers: int = 8,
        trans_nhead: int = 8,
        trans_dropout: float = 0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.cnn_out     = cnn_out

        convs = []
        in_ch = 1
        for _ in range(cnn_layers):
            k = cnn_kernel
            if k % 2 == 0:
                pad_left = k // 2
                pad_right = k // 2 - 1
                convs.append(nn.ConstantPad1d((pad_left, pad_right), 0))
                convs.append(nn.Conv1d(in_ch, cnn_out, k, padding=0))
            else:
                p = k // 2
                convs.append(nn.Conv1d(in_ch, cnn_out, k, padding=p))
            convs.append(nn.BatchNorm1d(cnn_out))
            convs.append(nn.ReLU())
            in_ch = cnn_out
        self.cnn = nn.Sequential(*convs)
        self.cnn_dropout = nn.Dropout(cnn_dropout)

        # Transformer encoder
        encoder = nn.TransformerEncoderLayer(
            d_model=cnn_out,
            nhead=trans_nhead,
            dim_feedforward=cnn_out*4,
            dropout=trans_dropout,
            activation='relu',
            batch_first=True,
        )
        self.cluster_transformer  = nn.TransformerEncoder(encoder, num_layers=trans_layers)
        self.position_transformer = nn.TransformerEncoder(encoder, num_layers=trans_layers)

    def forward(self, x: torch.Tensor):
        """
        x: (B, S, C)
        return:
          cl_out:  (B, S, cnn_out)
          pos_out: (B, S, cnn_out)
        """
        B, S, C = x.shape
        W = self.window_size
        pad = W // 2

        x_pad = F.pad(x.permute(0,2,1), (pad, pad), value=-1)  # (B, C, S+2*pad)

        win = x_pad.unfold(2, W, 1)
        win = win.permute(0,2,1,3)  # (B, S, C, W)
        bsz = B * S * C
        w = win.size(-1)
        cnn_input = win.reshape(bsz, 1, w)
        # CNN -> (bsz, cnn_out, w)
        feats = self.cnn(cnn_input)
        feats = self.cnn_dropout(feats)
        feats = feats.permute(0,2,1)  # (bsz, w, cnn_out)
        # reshape->(B, S, C, w, cnn_out)
        feats = feats.view(B, S, C, w, self.cnn_out)
        feats = feats.permute(0,1,3,2,4)  # (B, S, w, C, cnn_out)
        # cluster-wise
        center = w // 2
        cl_feat = feats[:,:,center,:,:].reshape(B*S, C, self.cnn_out)
        cl_out = self.cluster_transformer(cl_feat).mean(dim=1).view(B, S, self.cnn_out)
        # position-wise
        pos_feat = feats.mean(dim=3)  # (B, S, w, cnn_out)
        pos_in   = pos_feat.reshape(B*S, w, self.cnn_out)
        pos_out  = self.position_transformer(pos_in)[:,center,:].view(B, S, self.cnn_out)
        return cl_out, pos_out
