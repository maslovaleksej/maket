from models import *


def init_model(model_name):
    match (model_name):
        case "Resnet_v1_50":
            model = resnet_v1_50_CL_model
        case "Resnet_v1_101":
            model = resnet_v1_101_CL_model

        case _: model = resnet_v1_50_CL_model
    return model
