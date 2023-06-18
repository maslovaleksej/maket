import ins

dataset_name = "Flowers"
model_name = "Resnet_v1_50"
attack_name = "FGM"
defense_name = "SS"
batch_size = 16

model_obj = ins.Model_obj(
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

# model_obj.train_save(
#     dataset_name=dataset_name,
#     batch_size=batch_size,
#     epoch=3,
#     split=0.2,
#     device="GPU:0",
# )


model_obj.load(
    target_dataset_name=dataset_name,
    device="GPU:0",
)

model_obj.attack(
    attack_name=attack_name,
    batch_size=50,
    batch_nums=2,
    device="GPU:0",
    epsilon=[0.1, 0.15, 0.2, 0.6],
    save_images=True
)

model_obj.attack_def(
    attack_name=attack_name,
    batch_size=50,
    batch_nums=2,
    device="GPU:0",
    epsilon=[0.1, 0.15, 0.2, 0.6],
    save_images=True
)
