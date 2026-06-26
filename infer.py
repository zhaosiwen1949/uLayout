import argparse
import hydra
import torch
import numpy as np
import random
import sys
import pdb

from loguru import logger
from model.wrapper_ulayout import WrapperuLayout

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@hydra.main(config_path="config", config_name="main_config", version_base="1.3")
def main(cfg):
    
    cfg.id_exp = "ulayout" + "_" + cfg.pano_dataset + "_" + cfg.pp_dataset + "_" + cfg.mode
    fix_seed(cfg.model.seed)
    model = WrapperuLayout(cfg)

    model.prepare_for_validation_multi_dataset()
    model.set_valid_dataloader_custom(cfg.data_dir, mode=cfg.mode)
    model.plot_pano_custom(cfg.save_pred) # custom pano dataset has no label_cor ground truth.


if __name__ == "__main__":
    # Set the logger level to DEBUG
    logger.remove()
    logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD}</green> | <cyan>{function}</cyan>:<magenta>{line}</magenta> | <white>{message}</white>",
    colorize=True
    )
    logger.info("Starting the script...")
    main()  # Pass the args to main function
    
    