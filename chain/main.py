import os

import ins
from chain.disk_util import get_trained_model_save_path
from models import models


dataset_name = "Airplane"
attacks = ["PGD", "FGM", "BIM"]
defenses = ["SS", "JC", "FS"]

for model_obj in models:

    dir_path = get_trained_model_save_path(dataset_name=dataset_name, model_name=model_obj.file_name)
    print(model_obj.model_name)
    if not os.path.exists(dir_path):
        print (model_obj.file_name, "train")
        model_obj.train_save(
            dataset_name=dataset_name,
            batch_size=16,
            epoch=10,
            split=0.2,
            device="GPU:0",
        )


    # model_obj.load(
    #     target_dataset_name=dataset_name,
    #     device="GPU:0",
    # )

    # model_obj.attack(
    #     attack_name="FGM",
    #     batch_size=20,
    #     batch_nums=1,
    #     device="GPU:0",
    #     epsilon=[0.1, 0.15, 0.2, 0.6],
    #     save_images=True
    # )

    # model_obj.attack_def(
    #     attack_name="FGM",
    #     defense_name="SS",
    #     batch_size=20,
    #     batch_nums=1,
    #     device="GPU:0",
    #     epsilon=[0.1, 0.15, 0.2, 0.6],
    #     save_images=True
    # )
