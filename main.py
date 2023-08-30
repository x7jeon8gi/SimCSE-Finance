from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, random_split
from glob import glob
import os
from model import Fin_SimCSE
from dataset import UnsupDataset
from pathlib import Path
import yaml
import wandb
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

def get_root_dir():
    return Path(__file__).parent.parent

def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('SimCSE','configs', 'simcse_config.yaml'))
    return parser.parse_args()

def load_yaml_param_settings(yaml_fname: str):
    """
    :param yaml_fname: .yaml file that consists of hyper-parameter settings.
    """
    stream = open(yaml_fname, 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    return config

 
def train(config, train_data_loader, valid_data_loader, test_data_loader=None):
    
    project_name = 'SimCSE'
    group_name = config['train']['run_name']
    
    model = Fin_SimCSE(config, length_of_dataset=len(train_data_loader.dataset))
    wandb.init(project=project_name, name=None, config=config, group=group_name)
    wandb_logger = WandbLogger(project=project_name, name=None, config=config)
    wandb_logger.watch(model, log='all')
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor = 'val_epoch_loss',
        mode = 'min',
        dirpath = config['train']['saving_path'],
        filename = f'{group_name}'+'-{epoch}-{val_loss:.9f}',
    )
    
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=True,
                         callbacks=[LearningRateMonitor(logging_interval='epoch'), checkpoint_callback],
                         max_epochs = config['train']['epochs'],
                         accelerator='gpu',
                         strategy='ddp',
                         devices= config['train']['gpu_counts'] if torch.cuda.is_available() else None,
                         precision = config['train']['precision'],
                         )
    
    trainer.fit(model, 
                train_dataloaders =train_data_loader, 
                val_dataloaders = valid_data_loader)
    
    if test_data_loader is not None:
        trainer.test(test_dataloaders=test_data_loader)
    
    # save_model(model, config['train']['saving_path'], run_name)
    
    wandb.finish()
    
if __name__ == "__main__":
    
    args = load_args()
    config = load_yaml_param_settings(args.config)
    # not seed everything..
    
    dir = Path.cwd()
    
    train_dataset = UnsupDataset(mode='train', config=config)
    #test_dataset = UnsupDataset(mode='test', config=config)
    
    # random_split
    valid_size = int(len(train_dataset)*config['train']['valid_ratio'])
    train_dataset, valid_dataset = random_split(train_dataset, [len(train_dataset)-valid_size, valid_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['train']['num_workers'], pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=config['train']['num_workers'], pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=config['train']['num_workers'], pin_memory=True)
    
    train(config, train_loader, valid_loader)#, test_loader)    
    