from Model import Model

resnet_v1_50_CL_model = Model(
    type="CL",
    file_name="imagenet_resnet_v1_50_classification_5",
    model_name="Resnet_v1_50",
    input_size_x=224,
    input_size_y=224,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(batch_norm_momentum=0.997),
)

resnet_v1_101_CL_model = Model(
    type="CL",
    file_name="imagenet_resnet_v1_101_classification_5",
    model_name="Resnet_v1_101",
    input_size_x=224,
    input_size_y=224,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(batch_norm_momentum=0.997),
)
