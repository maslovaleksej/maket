import os
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import tensorflow as tf

from chain.const import DATA


def get_dir_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

def get_src_models_path(model_name):
    dir_path = DATA + "/Classification/Models/Src/" + model_name
    return dir_path

def get_datasets_path(dataset_name):
    dir_path = DATA + "/Classification/Datasets/" + dataset_name
    return dir_path


def get_trained_model_save_path(dataset_name, model_name):
    return DATA + "/Classification/Models/Trained/" + model_name + "__" + dataset_name



def save_adv_image(
        experiment_name,
        attack_name,
        dataset_name,
        model_name,
        eps,
        batch_count,
        labels,
        pred_labels,
        pred_adv_labels,
        adv_images):

    i = 0

    for adv_image in adv_images:

        label = labels[i]
        pred_label = np.argmax(pred_labels[i])
        pred_adv_label = np.argmax(pred_adv_labels[i])

        # dir_path = DATA + f"/Classification/Attacks/{attack_name}/{model_name}/{eps}/{dataset_name}/{label}"
        dir_path = DATA + f"/Classification/Experiments/{experiment_name}/{dataset_name}/{model_name}/ATTACK-{attack_name}/attacked_images"
        if not os.path.exists(dir_path): os.makedirs(dir_path)

        # plt.imsave(f'{dir_path}/img{batch_count:03d}-{i:05d}__{pred_label}-{pred_adv_label}.jpg', adv_images[i])
        plt.imsave(f'{dir_path}/{batch_count:03d}{i:03d}-{eps} ({label}-{pred_label}-{pred_adv_label}).jpg', adv_images[i])

        # img = Image.fromarray(np.asarray(np.clip(adv_image, 0, 255), dtype="uint8"), "L")
        # img.save(f'{dir_path}/img{batch_count:03d}-{i:05d}__{pred_label}-{pred_adv_label}.jpg')

        # out_img = Image.fromarray(adv_image, "RGB")
        # out_img.save("ycc.tif")

        i+=1


def save_adv_def_image(
        experiment_name,
        attack_name,
        defense_name,
        defense_eps,
        dataset_name,
        model_name,
        eps,
        batch_count,
        labels,
        pred_labels,
        pred_adv_labels,
        adv_images):

    i = 0

    for adv_image in adv_images:

        label = labels[i]
        pred_label = np.argmax(pred_labels[i])
        pred_adv_label = np.argmax(pred_adv_labels[i])

        # dir_path = DATA + f"/Classification/Attacks/{attack_name}_{defense_name}/{model_name}/{eps}/{dataset_name}/{label}"
        dir_path = DATA + f"/Classification/Experiments/{experiment_name}/{dataset_name}/{model_name}/ATTACK-{attack_name}/DEFENSE-{defense_name}"
        if not os.path.exists(dir_path): os.makedirs(dir_path)

        plt.imsave(f'{dir_path}/{batch_count:03d}{i:03d}-{eps}--{defense_eps} ({label}-{pred_label}-{pred_adv_label}).jpg', adv_images[i])

        # img = Image.fromarray(np.asarray(np.clip(adv_image, 0, 255), dtype="uint8"), "L")
        # img.save(f'{dir_path}/img{batch_count:03d}-{i:05d}__{pred_label}-{pred_adv_label}.jpg')

        # out_img = Image.fromarray(adv_image, "RGB")
        # out_img.save("ycc.tif")

        i+=1


def save_pred( experiment_name, attack_name, dataset_name, model_name, pred_acc_arr):
    dir_path = DATA + f"/Classification/Experiments/{experiment_name}/{dataset_name}/{model_name}/ATTACK-{attack_name}"

    if not os.path.exists(dir_path): os.makedirs(dir_path)

    with open(f"{dir_path}/attack_pred.txt", 'w') as fw:
        json.dump(pred_acc_arr, fw)


def save_pred_def( experiment_name, attack_name, dataset_name, defense_name, defense_eps, model_name, pred_acc_arr):
    # dir_path = DATA + f"/Classification/Attacks/{attack_name}/{model_name}"
    dir_path = DATA + f"/Classification/Experiments/{experiment_name}/{dataset_name}/{model_name}/ATTACK-{attack_name}"

    if not os.path.exists(dir_path): os.makedirs(dir_path)

    with open(f"{dir_path}/defense_{defense_name}_{defense_eps}_pred.txt", 'w') as fw:
        json.dump(pred_acc_arr, fw)



def save_history(dataset_name, model_name, history):
    dir_path = get_trained_model_save_path(dataset_name=dataset_name, model_name=model_name)
    if not os.path.exists(dir_path): os.makedirs(dir_path)

    f = open(dir_path + '/history.pickle', 'wb')
    pickle.dump(history, f)
    f.close()

    with open(dir_path + '/history.txt', 'w') as fw:
        json.dump(history, fw)




def save_model(dataset_name, model_name, model):
    dir_path = get_trained_model_save_path(dataset_name=dataset_name, model_name=model_name)
    if not os.path.exists(dir_path): os.makedirs(dir_path)

    # =====   H5  ====
    # model.save(dir_path + '/model.h5')


    # =====   TF ====
    tf.keras.models.save_model(model, dir_path)
    
    # =====   Pickle ====
    # f = open(dir_path + '/model.pickle', 'wb')
    # pickle.dump(model, f)
    # f.close()