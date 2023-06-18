import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from art.preprocessing.preprocessing import Preprocessor
from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier
from art.attacks.evasion import Wasserstein, FastGradientMethod, CarliniLInfMethod, BasicIterativeMethod, DeepFool, \
    ProjectedGradientDescent, ProjectedGradientDescentTensorFlowV2
from art.defences.preprocessor import SpatialSmoothing, FeatureSqueezing, JpegCompression
from keras.applications.resnet_v2 import preprocess_input
from sklearn.metrics import accuracy_score

import disk_util
import dataset_util
import const


class ResNet50Preprocessor(Preprocessor):

    def __call__(self, x, y=None):
        # print ("preprocessing...")
        return preprocess_input(x.copy()), y

    def estimate_gradient(self, x, gradient):
        return gradient[..., ::-1]


class Model_obj:
    def __init__(
            self,
            type,
            file_name,
            model_name,
            input_size_x,
            input_size_y,
            input_size_ch,
            min,
            max,
            build_shape,
            arguments,
    ):
        self.eps = None
        self.dataset_name = None
        self.history = None
        self.model = None
        self.model_type = type
        self.file_name = file_name
        self.model_name = model_name
        self.input_size_x = input_size_x
        self.input_size_y = input_size_y
        self.input_size_ch = input_size_ch
        self.min = min
        self.max = max
        self.build_shape = build_shape
        self.arguments = arguments
        self.image_shape = (input_size_x, input_size_y, input_size_ch)

    def train_save(self,
                   dataset_name,
                   batch_size,
                   epoch,
                   split,
                   device,
                   ):
        model_src_path = disk_util.get_src_models_path(self.file_name)
        dir_path = disk_util.get_datasets_path(dataset_name)

        train_ds = dataset_util.ImageGenerator(
            dir_path=dir_path,
            size_x=self.input_size_x,
            size_y=self.input_size_y,
            batch_size=batch_size,
            min=self.min,
            max=self.max,
            type="train",
            val_split=split
        )

        val_ds = dataset_util.ImageGenerator(
            dir_path=dir_path,
            size_x=self.input_size_x,
            size_y=self.input_size_y,
            batch_size=batch_size,
            min=self.min,
            max=self.max,
            type="val",
            val_split=split
        )

        with tf.device(device):
            self.model = tf.keras.Sequential([])
            self.model.add(hub.KerasLayer(model_src_path, trainable=True, arguments=self.arguments))

            self.model.add(tf.keras.layers.Dense(train_ds.class_nums * 4 * 4, activation='relu'))
            self.model.add(tf.keras.layers.Dense(train_ds.class_nums * 4, activation='relu'))

            # self.model.add(
            #     tf.keras.layers.Dense(self.train_ds.class_nums, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))

            self.model.add(
                tf.keras.layers.Dense(train_ds.class_nums)
            )

            self.model.build([None, self.input_size_x, self.input_size_y, self.input_size_ch])

            self.model.compile(
                optimizer=tf.keras.optimizers.SGD(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )

            self.model.summary()

            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

            self.history = self.model.fit(
                train_ds,
                epochs=epoch,
                steps_per_epoch=train_ds.batch_nums,
                validation_data=val_ds,
                validation_steps=val_ds.batch_nums,
                callbacks=[early_stopping_callback]
            ).history

            disk_util.save_model(
                dataset_name=dataset_name,
                model_name=self.model_name,
                model=self.model
            )
            disk_util.save_history(
                dataset_name=dataset_name,
                model_name=self.model_name,
                history=self.history
            )
        pass

    def load(self,
             target_dataset_name,
             device,
             ):

        with tf.device(device):
            model_path = disk_util.get_trained_model_save_path(dataset_name=target_dataset_name,
                                                               model_name=self.model_name)
            self.model = tf.keras.models.load_model(model_path, compile=True)
            self.model.compile()
            self.dataset_name = target_dataset_name
            # self.model.summary()

    def attack(
            self,
            attack_name,
            batch_size,
            batch_nums,
            device,
            save_images
    ):
        with tf.device(device):

            dir_path = disk_util.get_datasets_path(self.dataset_name)

            attack_ds = dataset_util.ImageGenerator(
                dir_path=dir_path,
                size_x=self.input_size_x,
                size_y=self.input_size_y,
                batch_size=batch_size,
                min=self.min,
                max=self.max,
                type="all",
            )

            pred_acc_arr = {"x": [], "y": [], "y_adv": []}

            classifier = TensorFlowV2Classifier(
                model=self.model,
                clip_values=(0, 1),
                nb_classes=attack_ds.class_nums,
                input_shape=self.image_shape,
                loss_object=tf.losses.SparseCategoricalCrossentropy(from_logits=True)
            )




            # tensorflowClassifier = TensorFlowV2Classifier(
            #     model=self.model,
            #     clip_values=(0, 1),
            #     nb_classes=attack_ds.class_nums,
            #     input_shape=self.image_shape,
            #     loss_object=tf.losses.SparseCategoricalCrossentropy(from_logits=True)
            # )
            #
            # kerasClassifier = KerasClassifier(
            #     model=self.model,
            #     clip_values=(0,1),
            #     use_logits=True,
            # )

            self.eps = [0, 0.1, 0.6, 0.9]

            for eps in self.eps:
                print("eps=", eps)
                match attack_name:
                    case "PGD":
                        attack = ProjectedGradientDescentTensorFlowV2(estimator=classifier, eps=eps)
                    case "FGM":
                        attack = FastGradientMethod(estimator=classifier, eps=eps)
                    case "BIM":
                        attack = BasicIterativeMethod(estimator=classifier, eps=eps)

                true_labels_arr = []
                pred_labels_arr = []
                pred_adv_labels_arr = []

                for i in range(batch_nums):
                    print("batch ", i)
                    images, labels = attack_ds.__getitem__(i)
                    adv_images = attack.generate(images)

                    pred_labels = self.model.predict(images)
                    pred_adv_labels = self.model.predict(adv_images)

                    disk_util.save_adv_image(
                        attack_name=attack_name,
                        dataset_name=self.dataset_name,
                        model_name=self.model_name,
                        eps=str(int(eps * 1000)),
                        batch_count=i,
                        labels=labels,
                        pred_labels=pred_labels,
                        pred_adv_labels=pred_adv_labels,
                        adv_images=adv_images
                    )

                    for label in labels:
                        true_labels_arr.append(label)

                    for label_ in pred_labels:
                        label = np.argmax(label_)
                        pred_labels_arr.append(label)

                    for label_ in pred_adv_labels:
                        label = np.argmax(label_)
                        pred_adv_labels_arr.append(label)

                    print(true_labels_arr)
                    print(pred_labels_arr)
                    print(pred_adv_labels_arr)

                pred_acc = accuracy_score(true_labels_arr, pred_labels_arr)
                pred_adv_acc = accuracy_score(true_labels_arr, pred_adv_labels_arr)

                print(f"for eps={eps} accuracy={pred_acc}")
                print(f"for eps={eps} adv_accuracy={pred_adv_acc}")
                print("\n")

                pred_acc_arr["x"].append(eps)
                pred_acc_arr["y"].append(pred_acc)
                pred_acc_arr["y_adv"].append(pred_adv_acc)

            print("pred_acc_arr", pred_acc_arr)

            disk_util.save_pred(
                attack_name=attack_name,
                dataset_name=self.dataset_name,
                model_name=self.model_name,
                pred_acc_arr=pred_acc_arr
            )

            print("\n\n")

    def attack_def(self,
                   attack_name,
                   def_name,
                   batch_size,
                   batch_nums,
                   device,
                   save_images
                   ):
        with tf.device(device):

            generator = dataset_util.Generator(
                dir_path=const.DATA + "/Classification/Datasets/Flowers/",
                size_x=self.input_size_x,
                size_y=self.input_size_y,
                batch_size=10,
                min=self.min,
                max=self.max
            )

            pred_acc_arr = {"x": [], "y": [], "y_adv": []}

            # classifier = TensorFlowV2Classifier(
            #     model=self.model,
            #     clip_values=(0, 1),
            #     nb_classes=generator.class_nums,
            #     input_shape=self.image_shape,
            #     loss_object=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            # )

            preprocessor = ResNet50Preprocessor()

            classifier = KerasClassifier(clip_values=(0, 255), model=self.model, preprocessing=preprocessor)

            jc = JpegCompression(clip_values=(0, 255), quality=10)
            classifier_def = KerasClassifier(preprocessing=preprocessor, preprocessing_defences=[jc],
                                             clip_values=(0, 255), model=self.model)


            self.eps = [0, 0.1, 0.6, 0.9]

            for eps in self.eps:
                print("eps=", eps)
                attack = FastGradientMethod(estimator=classifier, eps=eps)
                attack_def = FastGradientMethod(estimator=classifier_def, eps=eps)

                generator = dataset_util.Generator(
                    dir_path=const.DATA + "/Classification/Datasets/Flowers/",
                    size_x=self.input_size_x,
                    size_y=self.input_size_y,
                    batch_size=10,
                    min=self.min,
                    max=self.max
                )

                true_labels_arr = []
                pred_labels_arr = []
                pred_adv_labels_arr = []

                for i in range(batch_nums):
                    print("batch ", i)
                    gen = generator.get_batch_generator(batch_size=batch_size)
                    images, labels = next(gen)

                    adv_images = attack.generate(images)
                    adv_def_images = attack_def(images)

                    pred_labels = self.model.predict(images)
                    pred_adv_labels = self.model.predict(adv_images)

                    disk_util.save_adv_image(
                        attack_name=attack_name,
                        dataset_name=self.dataset_name,
                        model_name=self.model_name,
                        eps=str(int(eps * 1000)),
                        batch_count=i,
                        labels=labels,
                        pred_labels=pred_labels,
                        pred_adv_labels=pred_adv_labels,
                        adv_images=adv_images
                    )

                    for label in labels:
                        true_labels_arr.append(label)

                    for label_ in pred_labels:
                        label = np.argmax(label_)
                        pred_labels_arr.append(label)

                    for label_ in pred_adv_labels:
                        label = np.argmax(label_)
                        pred_adv_labels_arr.append(label)

                    print(true_labels_arr)
                    print(pred_labels_arr)
                    print(pred_adv_labels_arr)

                pred_acc = accuracy_score(true_labels_arr, pred_labels_arr)
                pred_adv_acc = accuracy_score(true_labels_arr, pred_adv_labels_arr)

                print(f"for eps={eps} accuracy={pred_acc}")
                print(f"for eps={eps} adv_accuracy={pred_adv_acc}")
                print("\n")

                pred_acc_arr["x"].append(eps)
                pred_acc_arr["y"].append(pred_acc)
                pred_acc_arr["y_adv"].append(pred_adv_acc)

            print("pred_acc_arr", pred_acc_arr)

            disk_util.save_pred(
                attack_name=attack_name,
                dataset_name=self.dataset_name,
                model_name=self.model_name,
                pred_acc_arr=pred_acc_arr
            )

            print("\n\n")
