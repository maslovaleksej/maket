import tensorflow as tf
from PIL import Image
import numpy as np
import os


class ImageGenerator(tf.keras.utils.Sequence):
    def __init__(
            self,
            dir_path,
            size_x=240,
            size_y=240,
            batch_size=16,
            min=0,
            max=1,
            type="both",
            val_split = 0.2
    ):

        self.size_x = size_x
        self.size_y = size_y
        self.batch_size = batch_size
        self.min = min
        self.max = max

        self.class_names = ""
        self.class_nums = 0
        self.total_size = 0


        class_names = []
        catalog = []
        size = 0

        if os.path.isdir(dir_path):
            for class_name in os.listdir(dir_path):
                path = os.path.join(dir_path, class_name)

                if os.path.isdir(path):
                    class_names.append(class_name)
                    files = os.listdir(path)
                    size += len(files)
                    files = [os.path.join(dir_path, class_name, file) for file in files]
                    catalog.append(files)

            self.class_names = class_names
            self.class_nums = len(class_names)
            self.catalog = catalog
            self.total_size = size

            # получаем массив размеров массивов каждого класса
            len_catalog_arrays = []
            for i in range(len(self.catalog)):
                len_catalog_arrays.append( len(self.catalog[i]) )

            # инициация процедуры
            class_num = 0
            class_step = 0
            count = 0
            image_vector = []
            label_vector = []

            # перебор каждого массива класса и выдергивание последовательно изображений каждого класса
            while count < self.total_size:
                if len_catalog_arrays[class_num] > class_step:
                    image_vector.append( self.catalog[class_num][class_step] )
                    label_vector.append( class_num )
                    count += 1

                class_num += 1

                if class_num >= self.class_nums:
                    class_num = 0
                    class_step += 1


            train_size = int(self.total_size * (1 - val_split))

            # на случай если не указано деление
            self.image_vector = image_vector
            self.label_vector = label_vector

            # указано деление
            if type=="train":
                self.image_vector = image_vector[:train_size]
                self.label_vector = label_vector[:train_size]

            if type=="val":
                self.image_vector = image_vector[train_size:]
                self.label_vector = label_vector[train_size:]

            # размер выборки в батчах
            self.batch_nums = len(self.image_vector) // self.batch_size



    def __len__(self):
            return self.batch_nums


    def __getitem__(self, index):
        # print ("get_item", index)

        # получаем имена файлов для пакета
        batch_file_paths = self.image_vector[index * self.batch_size : (index + 1) * self.batch_size]

        # получаем файлы для пакета
        batch_images = []
        for file_path in batch_file_paths:
            image = Image.open(file_path).convert('RGB')

            # преобразуем для обработки
            image_resized = image.resize((self.size_x, self.size_y))
            img = np.asarray(image_resized, dtype="float32")
            img = img / 255 * (self.max - self.min) + self.min
            # складываем
            batch_images.append(img)

        # получаем метки для пакета

        # for SparseCategoricalCrossentropy
        batch_labels = self.label_vector[index * self.batch_size : (index + 1) * self.batch_size]

        # for CategoricalCrossentropy
        # batch_labels = [tf.keras.utils.to_categorical(label, self.class_nums) for label in batch_labels]

        # преобразуем к формату =numpy=
        x = np.array(batch_images, dtype='float32')
        y = np.array(batch_labels)

        return x, y
        # return tf.convert_to_tensor(batch_images), tf.convert_to_tensor(batch_labels)




