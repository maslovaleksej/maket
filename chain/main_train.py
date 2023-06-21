import os

from chain.disk_util import get_trained_model_save_path
from models import models

dataset_names = ["Airplane", "Marvell", "Norb", "RsiCB256", "Flowers"]



# train models for datasets
for dataset_name in dataset_names:
    for model_obj in models:
        dir_path = get_trained_model_save_path(dataset_name=dataset_name, model_name=model_obj.model_name)
        print(dir_path)
        print(model_obj.model_name)

        if not os.path.exists(dir_path):
            print(model_obj.file_name, "train")
            model_obj.train_save(
                dataset_name=dataset_name,
                batch_size=4,
                epoch=20,
                split=0.2,
                device="GPU:0",
            )

