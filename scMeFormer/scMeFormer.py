import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import auroc, accuracy, f1_score, average_precision, matthews_corrcoef
from blocks import DNAModule, CpGModuleWithWindow
import torch.nn.functional as F
import ast

class ScMeFormer(pl.LightningModule):
    def __init__(
        self, C, n_cluster: int, ncell: int, segment_size: int=1024, RF: int = 2001, dna_channels: int = 4, 
        window_size: int = 101, lr: float = 0.000176, warmup_steps: int = 10000):
        super().__init__()
        self.save_hyperparameters()
        self.cluster_groups = []
        self.ncell = ncell
        with open(C, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx_list = ast.literal_eval(line)
                self.cluster_groups.append(idx_list)
        # DNA module -> (B, S, 256)
        self.dna_module = DNAModule(n_channels=dna_channels)
        # CpG module -> two outputs (B, S, 256)
        self.cpg_module = CpGModuleWithWindow(segment_size=segment_size,n_cluster=n_cluster,window_size=window_size)
        # Classification head
        self.classifier = nn.Sequential(nn.Linear(256 * 3, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, n_cluster))

    def process_batch(self, batch):
        x_windows, y_orig, C_matrix, ind_train = batch
        x_windows = x_windows.to(torch.long)
        dna_onehot = F.one_hot(x_windows.clamp(0,3), num_classes=4).float()
        y_orig = y_orig.to(torch.float)
        ind_train = ind_train.long()
        # torch.Size([1, 1024, 2001, 4]) torch.Size([1, 1024, 6]) torch.Size([1, 1024, 25]) torch.Size([1, 153, 2])
        return (dna_onehot, C_matrix), (y_orig, ind_train)
    
    def forward(self, dna_seq, cpg_matrix):
        # dna_seq: (B, S, dna_len, 4)
        # cpg_matrix: (B, S, S, n_cluster)
        dna_feat = self.dna_module(dna_seq)                       # (B, S, 256)
        cpg_cluster, cpg_position = self.cpg_module(cpg_matrix)    # each (B, S, 256)
        feat = torch.cat([dna_feat, cpg_cluster, cpg_position], dim=-1)  # (B, S, 768)
        logits = self.classifier(feat)   # (B, S)             
        return logits
    
    def training_step(self, batch, batch_idx):
        inputs, (y, ind_train) = self.process_batch(batch)
        y_hat = self(*inputs)
        cell2cluster = torch.empty(self.ncell, dtype=torch.long, device=y_hat.device)
        for c_idx, group in enumerate(self.cluster_groups):
            cell2cluster[group] = c_idx  
        y_hat = y_hat[:, :, cell2cluster]   # -> (B, S, n_cell)
        print('dsadasdads',y_hat.shape)

        y_hat = torch.diagonal(y_hat[:,ind_train[:,:,0], ind_train[:,:,1]]).reshape(-1)
        y = torch.diagonal(y[:,ind_train[:,:,0], ind_train[:,:,1]]).reshape(-1)
        
        loss = F.binary_cross_entropy_with_logits(y_hat, y-1)
        
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, (y, ind_train) = self.process_batch(batch)
        y_hat = self(*inputs)
        cell2cluster = torch.empty(self.ncell, dtype=torch.long, device=y_hat.device)
        for c_idx, group in enumerate(self.cluster_groups):
            cell2cluster[group] = c_idx  
        y_hat = y_hat[:, :, cell2cluster]   # -> (B, S, n_cell)
        #print('dsadasdads',y_hat.shape)
        
        y_hat = torch.diagonal(y_hat[:,ind_train[:,:,0], ind_train[:,:,1]]).reshape(-1)
        y = torch.diagonal(y[:,ind_train[:,:,0], ind_train[:,:,1]]).reshape(-1)
        return torch.stack((y_hat, y-1))
        
    
    def validation_epoch_end(self, validation_step_outputs):
        validation_step_outputs = torch.cat(validation_step_outputs,1)
        y_hat = validation_step_outputs[0]
        y = validation_step_outputs[1]
        
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        y_hat = torch.sigmoid(y_hat)
        self.log('val_loss', loss, sync_dist=True)
        y = y.to(torch.int)
        self.log('AUROC', auroc(y_hat, y, task='binary'), sync_dist=True)
        self.log('acc', accuracy(y_hat, y, task='binary'), sync_dist=True)
        self.log('f1', f1_score(y_hat, y, task='binary'), sync_dist=True)
        self.log('PRAUC', average_precision(y_hat, y, task='binary'), sync_dist=True)
        self.log('MCC', matthews_corrcoef(y_hat, y, num_classes=2, task='binary'), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lambd = lambda epoch: self.hparams.lr_decay_factor
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambd)
        return [optimizer], [lr_scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)
    
    def n_params(self):
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer += [('total', total_params)]
        return params_per_layer
    