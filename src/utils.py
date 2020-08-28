import random
import os
import numpy as np
import pandas as pd
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def summarize_submit(sub_list, experiment, filename='submission.csv'):
    res = pd.DataFrame()
    for i, path in enumerate(sub_list):
        sub = pd.read_csv(path)

        if i == 0:
            res['image_name'] = sub['image_name']
            res['target'] = sub['target']
        else:
            res['target'] += sub['target']
        os.remove(path)

    # min-max norm
    res['target'] -= res['target'].min()
    res['target'] /= res['target'].max()
    res.to_csv(filename, index=False)
    experiment.log_asset(file_data=filename, file_name=filename)
    os.remove(filename)

    return res
