from PIL import Image
import PIL
import os
import tensorflow as tf
import numpy as np

from glob import glob

import cv2
import matplotlib.pyplot as plt

import time
from IPython import display
import imageio
import sys

class InstanceNormalization(tf.keras.layers.Layer):
        """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

        def __init__(self, epsilon=1e-5):
            super(InstanceNormalization, self).__init__()
            self.epsilon = epsilon

        def build(self, input_shape):
            self.scale = self.add_weight(
                name='scale',
                shape=input_shape[-1:],
                initializer=tf.random_normal_initializer(1., 0.02),
                trainable=True)

            self.offset = self.add_weight(
                name='offset',
                shape=input_shape[-1:],
                initializer='zeros',
                trainable=True)

        def call(self, x):
            mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
            inv = tf.math.rsqrt(variance + self.epsilon)
            normalized = (x - mean) * inv
            return self.scale * normalized + self.offset

class CycleGAN():
    def __init__(self, from_checkpoint):
        self.BUFFER_SIZE = 1000
        self.BATCH_SIZE = 1
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
        self.EPOCHS = 2000
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.OUTPUT_CHANNELS = 3
        self.LAMBDA = 10

        self.do_preprocess = True
        self.from_checkpoint = from_checkpoint

        self.generator_draw = self.Generator(self.OUTPUT_CHANNELS, "draw", norm_type='instancenorm')
        self.generator_photo = self.Generator(self.OUTPUT_CHANNELS, "photo", norm_type='instancenorm')

        self.discriminator_photo = self.Discriminator("draw", norm_type='instancenorm', target=False)
        self.discriminator_draw = self.Discriminator("photo", norm_type='instancenorm', target=False)

        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


        self.generator_draw_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_photo_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_photo_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_draw_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.photo_dir = "./Cars/Teste_Carros/Photo_car"
        self.draw_dir = "./Cars/Teste_Carros/Draw_car"
        self.checkpoint_path = "./models/carroDesde1"

        self.ckpt = tf.train.Checkpoint(generator_draw=self.generator_draw,
                                        generator_photo=self.generator_photo,
                                        discriminator_photo=self.discriminator_photo,
                                        discriminator_draw=self.discriminator_draw,
                                        generator_draw_optimizer=self.generator_draw_optimizer,
                                        generator_photo_optimizer=self.generator_photo_optimizer,
                                        discriminator_photo_optimizer=self.discriminator_photo_optimizer,
                                        discriminator_draw_optimizer=self.discriminator_draw_optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=2)

    def resize(self, input_image, real_image, height, width):
        input_image = tf.image.resize(input_image, [height, width],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, real_image

    def random_crop(self, image):
        cropped_image = tf.image.random_crop(
            image, size=[self.IMG_HEIGHT, self.IMG_WIDTH, 3])

        return cropped_image

    # Transforma as imagens para [-1, 1]
    def normalize(self, image):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        return image

    def random_jitter(self, image):
        # Muda o tamanho para 286 x 286 x 3
        image = tf.image.resize(image, [286, 286],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Aleatóriamente corta a imagem para 256 x 256 x 3
        image = self.random_crop(image)

        # Espelha aleatóriamente
        image = tf.image.random_flip_left_right(image)

        return image

    def preprocess_image_train(self, image, label):
        image = self.random_jitter(image)
        image = self.normalize(image)
        return image

    def preprocess_image_test(self, image, label):
        image = self.normalize(image)
        return image

    def get_image(self, image_path, width, height, mode):
            image = Image.open(image_path)

            return np.array(image.convert(mode))

    def get_batch(self, image_files, width, height, mode):
            data_batch = np.array(
                [self.get_image(sample_file, width, height, mode) for sample_file in image_files]).astype(np.float32)

            # Make sure the images are in 4 dimensions
            if len(data_batch.shape) < 4:
                data_batch = data_batch.reshape(data_batch.shape + (1,))

            return data_batch

    def load_images(self, photo_dir, draw_dir):
        photos = []
        train_photos = []
        test_photos = []

        photos_name = np.array(glob(os.path.join(photo_dir, '*.png')))
        for photo in photos_name:
            np_photo = np.uint8(np.clip(cv2.imread(photo, 1), 0, 255))
            np_photo = cv2.cvtColor(np_photo, cv2.COLOR_BGR2RGB)
            photos.append(tf.convert_to_tensor(np_photo))

        n_train = 1

        for photo in photos:
            if n_train <= np.ceil(0.9 * len(photos)):
                train_photos.append(photo)
            else:
                test_photos.append(photo)
            n_train += 1


        train_photos_label = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor((np.zeros(len(train_photos), dtype=np.int64))))
        test_photos_label = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor((np.zeros(len(test_photos), dtype=np.int64))))

        train_photos = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(train_photos))
        test_photos = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(test_photos))

        train_photos = tf.data.Dataset.zip((train_photos, train_photos_label))
        test_photos = tf.data.Dataset.zip((test_photos, test_photos_label))

        draws = []
        train_draws = []
        test_draws = []

        draw_name = np.array(glob(os.path.join(draw_dir, '*.png')))
        for draw in draw_name:
            np_draw = np.uint8(np.clip(cv2.imread(draw, 1), 0, 255))
            np_draw = cv2.cvtColor(np_draw, cv2.COLOR_BGR2RGB)
            draws.append(tf.convert_to_tensor(np_draw))

        n_draw = 1

        for draw in draws:
            if n_draw <= np.ceil(0.9 * len(draws)):
                train_draws.append(draw)
            else:
                test_draws.append(draw)
            n_draw += 1

        train_draws_label = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor((np.zeros(len(train_draws), dtype=np.int64))))
        test_draws_label = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor((np.zeros(len(test_draws), dtype=np.int64))))

        train_draws = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(train_draws))
        test_draws = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(test_draws))

        train_draws = tf.data.Dataset.zip((train_draws, train_draws_label))
        test_draws = tf.data.Dataset.zip((test_draws, test_draws_label))


        train_photos = train_photos.map(
            self.preprocess_image_train, num_parallel_calls=self.AUTOTUNE).cache().shuffle(
            self.BUFFER_SIZE).batch(1)

        train_draws = train_draws.map(
            self.preprocess_image_train, num_parallel_calls=self.AUTOTUNE).cache().shuffle(
            self.BUFFER_SIZE).batch(1)

        test_photos = test_photos.map(
            self.preprocess_image_test, num_parallel_calls=self.AUTOTUNE).cache().shuffle(
            self.BUFFER_SIZE).batch(1)

        test_draws = test_draws.map(
            self.preprocess_image_test, num_parallel_calls=self.AUTOTUNE).cache().shuffle(
            self.BUFFER_SIZE).batch(1)

        return train_photos, test_photos, train_draws, test_draws

    def downsample(self, filters, size, name, norm_type='batchnorm', apply_norm=True):
        """Diminui um input.
            Conv2D => Batchnorm => LeakyRelu
            Args:
                filters: Número de Filtros
                size: Tamanho do filtro
                norm_type: Tipo de Normalização; 'batchnorm' ou 'instancenorm'.
                apply_norm: Se True, adiciona batchnorm layer
            Retorna:
                Downsample Sequential Model
        """
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False,
                                    name = name))

        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                result.add(tf.keras.layers.BatchNormalization())
            elif norm_type.lower() == 'instancenorm':
                result.add(InstanceNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(self, filters, size, name, norm_type='batchnorm', apply_dropout=False):
        """Aumenta um input.
        Conv2DTranspose => Batchnorm => Dropout => Relu
        Args:
            filters: Número de Filtros
            size: Tamanho do filtro
            norm_type: Tipo de Normalização; 'batchnorm' ou 'instancenorm'.
            apply_dropout: Se True, adiciona dropout layer
        Retorna:
            Upsample Sequential Model
        """

        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False,
                                            name = name))

        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def Generator(self, output_channels, genType, norm_type='batchnorm'):
        """Modified u-net generator model (https://arxiv.org/abs/1611.07004).
            Args:
                output_channels: Quantidade de Canis do Output
                norm_type: Tipo de Normalização; 'batchnorm' ou 'instancenorm'.
            Returna:
                Generator model
        """

        down_stack = [
            self.downsample(64, 4, ("gen_" + genType + "_01"), norm_type, apply_norm=False),  # (bs, 128, 128, 64)
            self.downsample(128, 4, ("gen_" + genType + "_02"), norm_type),  # (bs, 64, 64, 128)
            self.downsample(256, 4, ("gen_" + genType + "_03"), norm_type),  # (bs, 32, 32, 256)
            self.downsample(512, 4, ("gen_" + genType + "_04"), norm_type),  # (bs, 16, 16, 512)
            self.downsample(512, 4, ("gen_" + genType + "_05"), norm_type),  # (bs, 8, 8, 512)
            self.downsample(512, 4, ("gen_" + genType + "_06"), norm_type),  # (bs, 4, 4, 512)
            self.downsample(512, 4, ("gen_" + genType + "_07"), norm_type),  # (bs, 2, 2, 512)
            self.downsample(512, 4, ("gen_" + genType + "_08"), norm_type),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, 4, ("gen_" + genType + "_09"), norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
            self.upsample(512, 4, ("gen_" + genType + "_10"), norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
            self.upsample(512, 4, ("gen_" + genType + "_11"), norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
            self.upsample(512, 4, ("gen_" + genType + "_12"), norm_type),  # (bs, 16, 16, 1024)
            self.upsample(256, 4, ("gen_" + genType + "_13"), norm_type),  # (bs, 32, 32, 512)
            self.upsample(128, 4, ("gen_" + genType + "_14"), norm_type),  # (bs, 64, 64, 256)
            self.upsample(64, 4, ("gen_" + genType + "_15"), norm_type),  # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(
            output_channels, 4, strides=2,
            padding='same', kernel_initializer=initializer,
            activation='tanh',
            name=("gen_" + genType + "_output"))  # (bs, 256, 256, 3)

        concat = tf.keras.layers.Concatenate()

        inputs = tf.keras.layers.Input(shape=[None, None, 3], name = ("gen_" + genType + "_input"))
        x = inputs

        # Diminuindo
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Aumentando e estabelecendo os slips
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = concat([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def Discriminator(self, discType, norm_type='batchnorm', target=True):
        """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
        Args:
            norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
            target: Bool, indicating whether target image is an input or not.
        Returns:
            Discriminator model
        """

        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[None, None, 3], name=("disc_" + discType + "_input"))
        x = inp

        if target:
            tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
            x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

        down1 = self.downsample(64, 4, ("disc_" + discType + "_01"), norm_type, False)(x)  # (bs, 128, 128, 64)
        down2 = self.downsample(128, 4, ("disc_" + discType + "_02"), norm_type)(down1)  # (bs, 64, 64, 128)
        down3 = self.downsample(256, 4, ("disc_" + discType + "_03"), norm_type)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(
            512, 4, strides=1, kernel_initializer=initializer,
            use_bias=False, name=("disc_" + discType + "_5"))(zero_pad1)  # (bs, 31, 31, 512)

        if norm_type.lower() == 'batchnorm':
            norm1 = tf.keras.layers.BatchNormalization()(conv)
        elif norm_type.lower() == 'instancenorm':
            norm1 = InstanceNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(
            1, 4, strides=1,
            kernel_initializer=initializer, name=("disc_" + discType + "_output"))(zero_pad2)  # (bs, 30, 30, 1)

        if target:
            return tf.keras.Model(inputs=[inp, tar], outputs=last)
        else:
            return tf.keras.Model(inputs=inp, outputs=last)

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)

        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return self.LAMBDA * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss

    def generate_images(self, model, test_input, epoch):
        prediction = model(test_input, training=False)

        fig = plt.figure(figsize=(2.56, 2.56))

        plt.subplot(1, 1, 1)
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(prediction[0] * 0.5 + 0.5)
        plt.axis('off')

        # plt.savefig('./imgs/image_at_epoch_{:04d}.png'.format(epoch))
        return fig

    def train_step(self, real_photo, real_draw):
        # persistent is set to True because the tape is used more than once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator D traduz foto -> desenho
            # Generator P traduz desenho -> foto.

            # g = draw, f = photo, x = photo, y = draw

            self.fake_photo = self.generator_photo(real_draw, training=True)
            self.cycled_draw = self.generator_draw(self.fake_photo, training=True)

            self.fake_draw = self.generator_draw(real_photo, training=True)
            self.cycled_photo = self.generator_photo(self.fake_draw, training=True)

            # same_p and same_d are used for identity loss.
            self.same_photo = self.generator_photo(real_photo, training=True)
            self.same_draw = self.generator_draw(real_draw, training=True)

            self.disc_real_photo = self.discriminator_photo(real_photo, training=True)
            self.disc_real_draw = self.discriminator_draw(real_draw, training=True)

            self.disc_fake_photo = self.discriminator_photo(self.fake_photo, training=True)
            self.disc_fake_draw = self.discriminator_draw(self.fake_draw, training=True)

            # calculate the loss
            self.gen_photo_loss = self.generator_loss(self.disc_fake_photo)
            self.gen_draw_loss = self.generator_loss(self.disc_fake_draw)

            self.total_cycle_loss = self.calc_cycle_loss(real_photo, self.cycled_photo) + self.calc_cycle_loss(real_draw, self.cycled_draw)

            # Total generator loss = adversarial loss + cycle loss
            self.total_gen_photo_loss = self.gen_photo_loss + self.total_cycle_loss + self.identity_loss(real_photo, self.same_photo)
            self.total_gen_draw_loss = self.gen_draw_loss + self.total_cycle_loss + self.identity_loss(real_draw, self.same_draw)

            self.disc_photo_loss = self.discriminator_loss(self.disc_real_photo, self.disc_fake_photo)
            self.disc_draw_loss = self.discriminator_loss(self.disc_real_draw, self.disc_fake_draw)

        # Calculate the gradients for generator and discriminator
        self.generator_draw_gradients = tape.gradient(self.total_gen_draw_loss,
                                                        self.generator_draw.trainable_variables)

        self.generator_photo_gradients = tape.gradient(self.total_gen_photo_loss,
                                                        self.generator_photo.trainable_variables)

        self.discriminator_photo_gradients = tape.gradient(self.disc_photo_loss,
                                                            self.discriminator_photo.trainable_variables)

        self.discriminator_draw_gradients = tape.gradient(self.disc_draw_loss,
                                                            self.discriminator_draw.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_draw_optimizer.apply_gradients(zip(self.generator_draw_gradients,
                                                            self.generator_draw.trainable_variables))

        self.generator_photo_optimizer.apply_gradients(zip(self.generator_photo_gradients,
                                                            self.generator_photo.trainable_variables))

        self.discriminator_photo_optimizer.apply_gradients(zip(self.discriminator_photo_gradients,
                                                                self.discriminator_photo.trainable_variables))

        self.discriminator_draw_optimizer.apply_gradients(zip(self.discriminator_draw_gradients,
                                                                self.discriminator_draw.trainable_variables))

    def train(self, epochs = 0, draw = None):
        if self.from_checkpoint:
            # if a checkpoint exists, restore the latest checkpoint.
            # if self.ckpt_manager.latest_checkpoint:
            #     self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

            if np.all(draw) == None:
                train_photos, test_photos, train_draws, test_draws = self.load_images(self.photo_dir, self.draw_dir)
                sample_draw = next(iter(test_draws))
                photo = self.generate_images(self.generator_photo, sample_draw, epochs)
            else:
                tfImg = tf.data.Dataset.from_tensors(tf.convert_to_tensor(draw))
                label = tf.data.Dataset.from_tensors(tf.convert_to_tensor([0]))
                imgDS = tf.data.Dataset.zip((tfImg, label))

                imgDS = imgDS.map(
                self.preprocess_image_test, num_parallel_calls=self.AUTOTUNE).cache().shuffle( self.BUFFER_SIZE).batch(1)

                drawIter = next(iter(imgDS))
                photo = self.generate_images(self.generator_photo, drawIter, epochs)

            return photo

        else:
            print("Inicio do treino")
            if self.ckpt_manager.latest_checkpoint:
                self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

            for epoch in range(epochs):
                start = time.time()

                train_photos, test_photos, train_draws, test_draws = self.load_images(self.photo_dir, self.draw_dir)
                # sample_photo = next(iter(test_photos))
                sample_draw = next(iter(test_draws))
                print("Dataset importada")

                n = 0
                for image_x, image_y in tf.data.Dataset.zip((train_photos, train_draws)):
                    self.train_step(image_x, image_y)
                    if n % 1 == 0:
                        print ('.', end='')
                    n+=1
                # display.clear_output(wait=True)

                if (epoch + 1) % 15 == 0:
                    self.ckpt_save_path = self.ckpt_manager.save()
                    print ('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                        self.ckpt_save_path))

                print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                    time.time()-start))

if __name__ == "__main__":
    cycle = CycleGAN(False)
    cycle.train(cycle.EPOCHS)

    # def display_image(epoch_no):
    #     return PIL.Image.open('/content/imgs/image_at_epoch_{:04d}.png'.format(epoch_no))

    # display_image(1)

    # anim_file = 'dcgan.gif'

    # with imageio.get_writer(anim_file, mode='I') as writer:
    #     filenames = glob('/content/imgs/image*.png')
    #     filenames = sorted(filenames)
    #     last = -1
    #     for i,filename in enumerate(filenames):
    #         frame = 2*(i**0.5)
    #         if round(frame) > round(last):
    #             last = frame
    #         else:
    #             continue
    #         image = imageio.imread(filename)
    #         writer.append_data(image)
    #     image = imageio.imread(filename)
    #     writer.append_data(image)

    # import IPython
    # if IPython.version_info > (6,2,0,''):
    #     display.Image(filename=anim_file)
