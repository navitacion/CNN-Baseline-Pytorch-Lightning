import os
import glob
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.lightning import LightningSystem, DataModule
from src.models import ENet
from src.utils import seed_everything
from pytorch_lightning import Trainer
from comet_ml import Experiment

from pytorch_lightning.callbacks import ModelCheckpoint
from src.transforms import ImageTransform
from src.utils import summarize_submit

import warnings
warnings.filterwarnings('ignore')

# Config  ###########################
# Input Data
data_dir = './input'
# TTA
test_num = 20
# CV
cv = StratifiedKFold(n_splits=4, shuffle=True)


@hydra.main('config.yml')
def main(cfg: DictConfig):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    # Random Seed
    seed_everything(cfg.train.seed)

    # Model  ####################################################################
    net = ENet(model_name=cfg.train.model_name)
    transform = ImageTransform(img_size=cfg.data.img_size)

    # Comet.ml
    experiment = Experiment(api_key=cfg.comet_ml.api_key, project_name=cfg.comet_ml.project_name)
    # Log Parameters
    experiment.log_parameters(dict(cfg.exp))
    experiment.log_parameters(dict(cfg.data))
    experiment.log_parameters(dict(cfg.train))
    # Log Model Graph
    experiment.set_model_graph(str(net))

    # Lightning Module  #########################################################
    model = LightningSystem(net, cfg, experiment)
    datamodule = DataModule(data_dir, cfg, transform, cv)

    checkpoint_callback = ModelCheckpoint(
        filepath='./checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        prefix=cfg.exp.exp_name + '_'
    )

    trainer = Trainer(
        logger=False,
        max_epochs=cfg.train.epoch,
        checkpoint_callback=checkpoint_callback,
        gpus=1
            )

    # Train & Test  ############################################################
    # Train
    trainer.fit(model, datamodule=datamodule)
    experiment.log_metric('best_auc', model.best_auc)
    checkpoint_path = glob.glob(f'./checkpoint/{cfg.exp.exp_name}_*.ckpt')[0]
    experiment.log_asset(file_data=checkpoint_path)

    # Test
    for i in range(test_num):
        trainer.test(model)

    # Submit
    sub_list = glob.glob(f'submission_{cfg.exp.exp_name}*.csv')
    _ = summarize_submit(sub_list, experiment, filename=f'sub_{cfg.exp.exp_name}.csv')

    # oof
    oof_dataset = datamodule.oof_dataset
    oof_dataloader = DataLoader(oof_dataset, batch_size=cfg.train.batch_size, pin_memory=False,
                                shuffle=False, drop_last=False)
    for i in range(10):
        trainer.test(model, test_dataloaders=oof_dataloader)

    # Submit
    sub_list = glob.glob('submission*.csv')
    _ = summarize_submit(sub_list, experiment, filename=f'oof_{cfg.exp.exp_name}.csv')


if __name__ == '__main__':
    main()
