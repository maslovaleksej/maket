from chain.models import models

dataset_names = ["Airplane", "Marvell", "Norb", "RsiCB256", "Flowers"]
experiment_name = "test_robustness"
save_images = True

# defenses = ["SS", "JC", "FS", ]
#
# # train models for datasets
for dataset_name in dataset_names:
    for model_obj in models:

        model_obj.load(
            target_dataset_name=dataset_name,
            device="GPU:0",
        )

        model_obj.attack(
            experiment_name = experiment_name,
            attack_name="FGM",
            batch_size=20,
            batch_nums=1,
            device="GPU:0",
            epsilon=[0, 0.01, 0.1, 0.15, 0.2, 0.6],
            save_images=save_images
        )
        model_obj.attack_defense(
            experiment_name=experiment_name,
            attack_name="FGM",
            epsilon=[0, 0.01, 0.1, 0.15, 0.2, 0.6],
            defense_name="SS",
            defense_eps=6,
            batch_size=20,
            batch_nums=1,
            device="GPU:0",
            save_images=save_images
        )
        model_obj.attack_defense(
            experiment_name=experiment_name,
            attack_name="FGM",
            epsilon=[0, 0.01, 0.1, 0.15, 0.2, 0.6],
            defense_name="SS",
            defense_eps=9,
            batch_size=20,
            batch_nums=1,
            device="GPU:0",
            save_images=save_images
        )



        model_obj.attack(
            experiment_name = experiment_name,
            attack_name="PGD",
            batch_size=20,
            batch_nums=1,
            device="GPU:0",
            epsilon=[0.01, 0.1, 0.15, 0.2, 0.6],
            save_images=save_images
        )

        model_obj.attack(
            experiment_name=experiment_name,
            attack_name="BIM",
            batch_size=20,
            batch_nums=1,
            device="GPU:0",
            epsilon=[0.01, 0.1, 0.15, 0.2, 0.6],
            save_images=save_images
        )




print ("done")