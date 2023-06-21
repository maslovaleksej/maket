import os

from chain.const import DATA
from chain.models import models

dataset_names = ["Airplane", "Marvell", "Norb", "RsiCB256", "Flowers"]
dataset_names = ["Airplane", "Marvell", "Norb"]

experiment_name = "test_robustness"
save_images = False
batch_size = 4 # 10 уже не проходит на GPU 12gb for VIT
batch_nums = 1
device = "GPU:0"

attacks = [
    {"name": "PGD", "eps": [0.001, 0.002, 0.01, 0.02, 0.1, 0.2]},
    {"name": "FGM", "eps": [0.001, 0.002, 0.01, 0.02, 0.1, 0.2]},
    {"name": "BIM", "eps": [0.001, 0.002, 0.01, 0.02, 0.1, 0.2]},
]

defenses = [
    {"name": "JC", "eps": [8, 10, 12]},
    {"name": "SS", "eps": [4, 6, 8]},
    {"name": "FS", "eps": [7, 8, 10]},
]

for dataset_name in dataset_names:
    print("===================================")
    print("dataset_name", dataset_name)
    print("===================================")

    for model_obj in models:
        print("-----------------------------------")
        print("model_name", model_obj.model_name)
        print("-----------------------------------")

        model_obj.load(
            target_dataset_name=dataset_name,
            device="GPU:0",
        )

        #  =============== attack ==================

        for attack in attacks:
            print("--------------------------------------")
            print("attack", attack["name"])
            print("epslinons", attack["eps"])
            print("--------------------------------------")

            dir_path = DATA + f"/Classification/Experiments/{experiment_name}/{dataset_name}/{model_obj.model_name}/ATTACK-{attack['name']}"
            if not os.path.exists(dir_path):

                model_obj.attack(
                    experiment_name=experiment_name,
                    attack_name=attack["name"],
                    epsilons=attack["eps"],
                    batch_size=batch_size,
                    batch_nums=batch_nums,
                    device=device,
                    save_images=save_images
                )

            # for defense in defenses:
            #     print("")
            #     print("attack", attack["name"])
            #     print("attack epsilons", attack["eps"])
            #     print("defense", defense["name"])
            #     print("")
            #
            #     for defense_epsilon in defense["eps"]:
            #         print("")
            #         print("attack", attack["name"])
            #         print("attack epsilons", attack["eps"])
            #         print("defense", defense["name"])
            #         print("defense epsilon", defense_epsilon)

            # -------------- defense -------------------

            # model_obj.attack_defense(
            #     experiment_name=experiment_name,
            #     attack_name=attack["name"],
            #     eps=attack_epsilon,
            #     defense_name=defense["name"],
            #     defense_eps=defense_epsilon,
            #     batch_size=batch_size,
            #     batch_nums=batch_nums,
            #     device=device,
            #     save_images=save_images
            # )

print("done")
