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

# def get_trained_datasets_path():
#     dir_path = const.DATA + "/Classification/Models/Trained"
#     res_path = []
#     for path in os.listdir(dir_path):
#         path_to_file = os.path.join(dir_path, path)
#         if os.path.isdir(path_to_file):
#             if path[0] != '_':
#                 res_path.append(path_to_file)
#     return res_path

# def get_trained_models_path(dataset_name, model_name):
#     dir_path = const.DATA + "/Classification/Models/Trained/" + dataset_name
#     res_path = []
#     for path in os.listdir(dir_path):
#         path_to_file = os.path.join(dir_path, path)
#         if os.path.isdir(path_to_file):
#             if path[0] != '_':
#                 res_path.append(path_to_file)
#     return res_path



def save_adv_image(
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

        dir_path = DATA + f"/Classification/Attacks/{attack_name}/{model_name}/{eps}/{dataset_name}/{label}"
        if not os.path.exists(dir_path): os.makedirs(dir_path)

        plt.imsave(f'{dir_path}/img{batch_count:03d}-{i:05d}__{pred_label}-{pred_adv_label}.jpg', adv_images[i])

        # img = Image.fromarray(np.asarray(np.clip(adv_image, 0, 255), dtype="uint8"), "L")
        # img.save(f'{dir_path}/img{batch_count:03d}-{i:05d}__{pred_label}-{pred_adv_label}.jpg')

        # out_img = Image.fromarray(adv_image, "RGB")
        # out_img.save("ycc.tif")

        i+=1


def save_pred(attack_name, dataset_name, model_name, pred_acc_arr):
    dir_path = DATA + f"/Classification/Attacks/{attack_name}/{model_name}"

    if not os.path.exists(dir_path): os.makedirs(dir_path)

    with open(f"{dir_path}/{dataset_name}_pred.txt", 'w') as fw:
        json.dump(pred_acc_arr, fw)
    pass


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