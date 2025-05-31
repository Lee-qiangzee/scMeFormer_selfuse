import torch
import numpy as np
import random
import pytorch_lightning as pl
import math
import ast

def sample_from_cdf(cdf, n):
    return cdf['x'][(torch.rand(n, 1) < cdf['y']).int().argmax(-1)]

class ScMeFormerDataModule(pl.LightningDataModule):
    def __init__(self, X, y, pos, C, segment_size=1024, RF=2001, fracs=[1,0,0],
                 mask_perc=0.25, mask_random_perc=0.2,
                 resample_cells=None, resample_cells_val=None,
                 val_keys=None, test_keys=None,
                 batch_size=1, n_workers=4):
        
        assert len(fracs)==3,'length of fractions should be 3 for train/val/test'
        assert sum(fracs)==1, 'Sum of train/val/test fractions should be one.'
        assert val_keys is None or type(val_keys) is list, 'val_keys should be None or list'
        assert test_keys is None or type(test_keys) is list, 'test_keys should be None or list'
        if val_keys is not None and test_keys is not None:
            assert set(val_keys) & set(test_keys) == set(), 'No overlap allowed between val_keys & test_keys'
        super().__init__()
        
        self.X = X
        self.y = y
        self.pos = pos
        self.Cinfo = C
        self.segment_size = segment_size
        self.RF = RF; self.RF2 = int((RF-1)/2)
        self.fracs = fracs
        self.val_keys = val_keys
        self.test_keys = test_keys
        self.mask_perc = mask_perc
        self.mask_random_perc = mask_random_perc
        self.bsz = batch_size
        self.nw = n_workers
        self.resample = resample_cells
        self.resample_val = resample_cells_val
        
    def setup(self, stage):
        train = []; val = []; test = []
        
        for chr_name in self.y.keys():
            y_temp = self.y[chr_name]
            X_temp = self.X[chr_name]
            pos_temp = self.pos[chr_name]
            
            if 'numpy' in str(type(X_temp)):
                X_temp = torch.from_numpy(X_temp)
                y_temp = torch.from_numpy(y_temp)
                pos_temp = torch.from_numpy(pos_temp)
                
               
            X_temp = torch.cat((torch.full((self.RF2,),4, dtype=torch.int8), X_temp,
                                torch.full((self.RF2,),4, dtype=torch.int8)))
            pos_temp = pos_temp.clone() + self.RF2

            # mask gaps, deleting the parts of the genome where no CpG sites are labeled.
            mask = torch.ones_like(X_temp, dtype=torch.bool)
            for e, b in zip(pos_temp[1:][pos_temp[1:] - pos_temp[:-1] > self.RF],
                            pos_temp[:-1][pos_temp[1:] - pos_temp[:-1] > self.RF]):
                mask[torch.arange(b+self.RF2+1,e-self.RF2)] = False

            tmp = torch.zeros_like(X_temp, dtype=torch.int8)
            tmp[pos_temp.to(torch.long)] = 1
            tmp = tmp[mask]
            indices = torch.where(tmp)[0]
            X_temp = X_temp[mask]


            # skip the chromosome if it has less than segment_size CpG sites
            n_pos = len(pos_temp)
            if n_pos < self.segment_size:
                continue
            
            # prepare cuts that segment the genome & labels
            cuts_ = torch.arange(0,n_pos-self.segment_size+1,self.segment_size)
            cuts = torch.tensor([(indices[i],indices[i+self.segment_size-1]) for i in cuts_])
            cuts_ = torch.cat((cuts_, torch.tensor([n_pos-self.segment_size])))
            cuts = torch.cat((cuts, torch.tensor([(indices[-self.segment_size], indices[-1])])))
            batched_temp=[(X_temp[max(srt-self.RF2,0):stp+1+self.RF2],
                           y_temp[i:i+self.segment_size],
                           indices[i:i+self.segment_size]-indices[i]+self.RF2, 
                           pos_temp[i:i+self.segment_size]-pos_temp[i]) for i, (srt, stp) in zip(cuts_, cuts)]


            if self.val_keys is not None and chr_name in self.val_keys:
                val += batched_temp
            elif self.test_keys is not None and chr_name in self.test_keys:
                test += batched_temp
            elif self.fracs != [1,0,0]:
                random.shuffle(batched_temp)
                splits = np.cumsum(np.round(np.array(self.fracs)*len(batched_temp)).astype('int'))
                train += batched_temp[:splits[0]]
                val += batched_temp[splits[0]:splits[1]]
                test += batched_temp[splits[1]:]
            else:
                train += batched_temp

        self.train = ScMeFormerDataset(train, C = self.Cinfo, RF=self.RF,
                                           mask_percentage=self.mask_perc, 
                                           mask_random_percentage=self.mask_random_perc,
                                           resample_cells=self.resample)
        
        self.val = ScMeFormerDataset(val, C = self.Cinfo, RF=self.RF,
                                         mask_percentage=self.mask_perc, 
                                         mask_random_percentage=self.mask_random_perc,
                                         resample_cells=self.resample_val)

        self.test = test
        
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, num_workers=self.nw,
                                           batch_size=self.bsz, shuffle=True,
                                           pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, num_workers=self.nw,
                                           batch_size=self.bsz, shuffle=False,
                                           pin_memory=True)
    
class ScMeFormerDataset(torch.utils.data.Dataset):
    def __init__(self, split, C, RF=2001, mask_percentage=0.25,
                 mask_random_percentage=0.20, resample_cells=None):
        self.split = split
        
        RF2 = int((RF-1)/2)

        self.Cinfo = C
        self.cluster_groups = []
        with open(C, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx_list = ast.literal_eval(line)
                self.cluster_groups.append(idx_list)
        self.n_clusters = len(self.cluster_groups)

        self.r = torch.arange(-RF2, RF2+1)
        self.k = RF
        
        # make a CDF of label distribution to sample from in randomizing:
        s = torch.stack([s[1] for s in split])
        s = s[s != -1]
        indices = torch.randperm(s.shape[0])[:2500]
        self.cdf = {'x': s[indices].sort().values, 'y': torch.linspace(0, 1, 2500)}
        
        self.mp = mask_percentage
        self.mrp = mask_random_percentage
        
        self.resample = resample_cells
        
        
    def __len__(self):
        return len(self.split)
    
    def __getitem__(self, index):
        x, y, ind, pos = self.split[index] # -1: unknown, 0: not methylated, 1: methylated
        
        x_windows = x[ind.unsqueeze(1).repeat(1,self.k)+self.r]

        y_orig = y+1
        seqlen, n_rep = y_orig.size()
        y_masked = y_orig.clone()
        
        nonzeros = y_masked.nonzero(as_tuple=False)
        n_permute = min(int(seqlen*self.mp), nonzeros.size(0))
        
        if self.mrp:
            n_mask, n_random = int(n_permute*(1-self.mrp)), math.ceil(n_permute*self.mrp)
            perm = torch.randperm(nonzeros.size(0))[:n_permute]
            nonzeros = nonzeros[perm]
            mask, rand = torch.split(nonzeros,[n_mask,n_random])
            
            y_masked[mask[:,0],mask[:,1]] = 0
            y_masked[rand[:,0],rand[:,1]] = sample_from_cdf(self.cdf, n_random)+1
        else:
            perm = torch.randperm(nonzeros.size(0))[:n_permute]
            nonzeros = nonzeros[perm]
            
            y_masked[nonzeros[:,0],nonzeros[:,1]] = 0
        # print('ssssss',y_masked.shape)      segment_size, ncell
        C_matrix = torch.full((seqlen, self.n_clusters), -1.0, dtype=torch.float)
        for c, group in enumerate(self.cluster_groups):
            if len(group) == 0:
                continue
            block = y_masked[:, group]            # (seqlen, cluster_size)
            mask = (block >= 0)                
            count = mask.sum(dim=1)    
            sum_vals = (block * mask).sum(dim=1).float()
            valid = count > 0
            C_matrix[valid, c] = sum_vals[valid] / count[valid].float()
        return x_windows, y_orig, C_matrix, nonzeros