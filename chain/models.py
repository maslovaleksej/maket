from chain.ins import Model_obj

models = []

model = Model_obj(
    type="FV",
    file_name="efficientnet_b0_feature-vector_1",
    model_name="efficientnet_b0",
    input_size_x=224,
    input_size_y=224,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(),
)
models.append(model)


model = Model_obj(
    type="FV",
    file_name="efficientnet_b3_feature-vector_1",
    model_name="efficientnet_b3",
    input_size_x=300,
    input_size_y=300,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(),
)
models.append(model)


model = Model_obj(
    type="FV",
    file_name="imagenet_inception_resnet_v2_feature_vector_5",
    model_name="inception_resnet_v2",
    input_size_x=299,
    input_size_y=299,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(batch_norm_momentum=0.997),
)
models.append(model)


model = Model_obj(
    type="FV",
    file_name="imagenet_inception_v1_feature_vector_5",
    model_name="inception_v1",
    input_size_x=224,
    input_size_y=224,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(batch_norm_momentum=0.997),
)
models.append(model)


model = Model_obj(
    type="FV",
    file_name="imagenet_inception_v2_feature_vector_5",
    model_name="inception_v2",
    input_size_x=224,
    input_size_y=224,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(batch_norm_momentum=0.997),
)
models.append(model)


model = Model_obj(
    type="FV",
    file_name="imagenet_inception_v3_feature_vector_5",
    model_name="inception_v3",
    input_size_x=299,
    input_size_y=299,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(batch_norm_momentum=0.997),
)
models.append(model)



model = Model_obj(
    type="FV",
    file_name="imagenet_mobilenet_v1_100_224_feature_vector_5",
    model_name="mobilenet_v1_100",
    input_size_x=224,
    input_size_y=224,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(batch_norm_momentum=0.997),
)
models.append(model)


model = Model_obj(
    type="FV",
    file_name="imagenet_mobilenet_v2_140_224_feature_vector_5",
    model_name="mobilenet_v2_140",
    input_size_x=224,
    input_size_y=224,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(batch_norm_momentum=0.997),
)
models.append(model)


model = Model_obj(
    type="FV",
    file_name="imagenet_mobilenet_v3_large_100_224_feature_vector_5",
    model_name="mobilenet_v3_large_100",
    input_size_x=224,
    input_size_y=224,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(batch_norm_momentum=0.997),
)
models.append(model)




model = Model_obj(
    type="CL",
    file_name="imagenet_resnet_v1_50_classification_5",
    model_name="resnet_v1_50",
    input_size_x=224,
    input_size_y=224,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(batch_norm_momentum=0.997),
)
models.append(model)


model = Model_obj(
    type="CL",
    file_name="imagenet_resnet_v1_101_classification_5",
    model_name="resnet_v1_101",
    input_size_x=224,
    input_size_y=224,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(batch_norm_momentum=0.997),
)
models.append(model)


model = Model_obj(
    type="CL",
    file_name="imagenet_resnet_v1_152_classification_5",
    model_name="resnet_v1_152",
    input_size_x=224,
    input_size_y=224,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(batch_norm_momentum=0.997),
)
models.append(model)


model = Model_obj(
    type="CL",
    file_name="imagenet_resnet_v2_50_classification_5",
    model_name="resnet_v2_50",
    input_size_x=224,
    input_size_y=224,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(batch_norm_momentum=0.997),
)
models.append(model)


model = Model_obj(
    type="CL",
    file_name="imagenet_resnet_v2_101_classification_5",
    model_name="resnet_v2_101",
    input_size_x=224,
    input_size_y=224,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(batch_norm_momentum=0.997),
)
models.append(model)


model = Model_obj(
    type="CL",
    file_name="imagenet_resnet_v2_152_classification_5",
    model_name="resnet_v2_152",
    input_size_x=224,
    input_size_y=224,
    input_size_ch=3,
    min=0,
    max=1,
    build_shape=True,
    arguments=dict(batch_norm_momentum=0.997),
)
models.append(model)



model = Model_obj(
    type="FV",
    file_name="vit_b8_fe_1",
    model_name="vit_b8",
    input_size_x=224,
    input_size_y=224,
    input_size_ch=3,
    min=-1,
    max=1,
    build_shape=True,
    arguments=dict(),
)
models.append(model)


# model = Model_obj(
#     type="FV",
#     file_name="vit_b16_fe_1",
#     model_name="vit_b16",
#     input_size_x=224,
#     input_size_y=224,
#     input_size_ch=3,
#     min=-1,
#     max=1,
#     build_shape=True,
#     arguments=dict(),
# )
# models.append(model)


# model = Model_obj(
#     type="FV",
#     file_name="vit_b32_fe_1",
#     model_name="vit_b32",
#     input_size_x=224,
#     input_size_y=224,
#     input_size_ch=3,
#     min=-1,
#     max=1,
#     build_shape=True,
#     arguments=dict(),
# )
# models.append(model)
