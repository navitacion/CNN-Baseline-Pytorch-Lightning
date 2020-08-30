import pandas as pd
import itertools
import glob
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
from .datasets import ImageDataset

from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.metrics.classification import AUROC


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, cfg, transform, cv):
        super(DataModule, self).__init__()
        self.cfg = cfg
        self.data_dir = data_dir
        self.transform = transform
        self.cv = cv


    def prepare_data(self):
        # Prepare Data
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.test_dir = os.path.join(self.data_dir, 'test')
        self.df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.test = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))


    def setup(self, stage=None):
        # Split Train, Validation
        self.df['fold'] = -1
        for i, (trn_idx, val_idx) in enumerate(self.cv.split(self.df, self.df['target'])):
            self.df.loc[val_idx, 'fold'] = i
        fold = self.cfg.train.fold
        train = self.df[self.df['fold'] != fold].reset_index(drop=True)
        val = self.df[self.df['fold'] == fold].reset_index(drop=True)

        self.train_dataset = ImageDataset(train, self.train_dir, self.transform, phase='train')
        self.val_dataset = ImageDataset(val, self.train_dir, self.transform, phase='val')
        self.test_dataset = ImageDataset(self.test, self.test_dir, self.transform, phase='test')
        self.oof_dataset = ImageDataset(self.df, self.train_dir, self.transform, phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.train.batch_size,
                          pin_memory=True,
                          sampler=RandomSampler(self.train_dataset), drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.train.batch_size,
                          pin_memory=True,
                          sampler=SequentialSampler(self.val_dataset), drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.cfg.train.batch_size,
                          pin_memory=False,
                          shuffle=False, drop_last=False)



class LightningSystem(pl.LightningModule):
    def __init__(self, net, cfg, experiment):
        super(LightningSystem, self).__init__()
        self.net = net
        self.cfg = cfg
        self.experiment = experiment
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_loss = 1e+9
        self.best_auc = None
        self.epoch_num = 0


    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.parameters(), lr=self.cfg.train.lr, weight_decay=2e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.train.epoch, eta_min=0)

        return [self.optimizer], [self.scheduler]

    def forward(self, x):
        return self.net(x)

    def step(self, batch):
        inp, label = batch
        out = self.forward(inp)
        loss = self.criterion(out, label.unsqueeze(1))

        return loss, label, torch.sigmoid(out)

    def training_step(self, batch, batch_idx):
        loss, label, logits = self.step(batch)

        logs = {'train/loss': loss.item()}
        # batch_idx + Epoch * Iteration
        step = batch_idx
        self.experiment.log_metrics(logs, step=step)

        return {'loss': loss, 'logits': logits, 'labels': label}

    def validation_step(self, batch, batch_idx):
        loss, label, logits = self.step(batch)

        val_logs = {'val/loss': loss.item()}
        # batch_idx + Epoch * Iteration
        step = batch_idx
        self.experiment.log_metrics(val_logs, step=step)

        return {'val_loss': loss, 'logits': logits.detach(), 'labels': label.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        LOGITS = torch.cat([x['logits'] for x in outputs])
        LABELS = torch.cat([x['labels'] for x in outputs])

        # Skip Sanity Check
        auc = AUROC()(pred=LOGITS.detach(), target=LABELS.detach()) if LABELS.float().mean() > 0 else 0.5
        logs = {'val/epoch_loss': avg_loss.item(), 'val/epoch_auc': auc}
        # Log loss, auc
        self.experiment.log_metrics(logs, step=self.epoch_num)
        # Update Epoch Num
        self.epoch_num += 1

        # Save Weights
        if self.best_loss > avg_loss:
            self.best_loss = avg_loss
            filename = f'{self.cfg.exp.exp_name}_epoch_{self.epoch_num}_loss_{self.best_loss:.3f}_auc_{auc:.3f}.pth'
            torch.save(self.net.state_dict(), filename)
            self.experiment.log_model(name=filename, file_or_folder='./'+filename)
            os.remove(filename)
            self.best_auc = auc

        return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        inp, img_name = batch
        out = self.forward(inp)
        logits = torch.sigmoid(out)

        return {'preds': logits, 'image_names': img_name}

    def test_epoch_end(self, outputs):
        PREDS = torch.cat([x['preds'] for x in outputs]).reshape((-1)).detach().cpu().numpy()
        # [tuple, tuple]
        IMG_NAMES = [x['image_names'] for x in outputs]
        # [list, list]
        IMG_NAMES = [list(x) for x in IMG_NAMES]
        IMG_NAMES = list(itertools.chain.from_iterable(IMG_NAMES))

        res = pd.DataFrame({
            'image_name': IMG_NAMES,
            'target': PREDS
        })
        try:
            res['target'] = res['target'].apply(lambda x: x.replace('[', '').replace(']', ''))
        except:
            pass
        N = len(glob.glob(f'submission_{self.cfg.exp.exp_name}*.csv'))
        filename = f'submission_{self.cfg.exp.exp_name}_{N}.csv'
        res.to_csv(filename, index=False)

        return {'res': res}
