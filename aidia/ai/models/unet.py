import tensorflow as tf


class UNet(tf.keras.Model):
    """U-Net basic model."""
    def __init__(self, num_classes):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder(num_classes)

    def call(self, x):
        z1, z2, z3, z4_dropout, z5_dropout = self.enc(x)
        y = self.dec(z1, z2, z3, z4_dropout, z5_dropout)
        return y
    
class Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Network
        self.block1_conv1 = tf.keras.layers.Conv2D(
            64, (3, 3), name='block1_conv1', activation='relu', padding='same')
        self.block1_conv2 = tf.keras.layers.Conv2D(
            64, (3, 3), name='block1_conv2', padding='same')
        self.block1_bn = tf.keras.layers.BatchNormalization()
        self.block1_act = tf.keras.layers.ReLU()
        self.block1_pool = tf.keras.layers.MaxPooling2D(
            (2, 2), strides=None, name='block1_pool')

        self.block2_conv1 = tf.keras.layers.Conv2D(
            128, (3, 3), name='block2_conv1', activation='relu', padding='same')
        self.block2_conv2 = tf.keras.layers.Conv2D(
            128, (3, 3), name='block2_conv2', padding='same')
        self.block2_bn = tf.keras.layers.BatchNormalization()
        self.block2_act = tf.keras.layers.ReLU()
        self.block2_pool = tf.keras.layers.MaxPooling2D(
            (2, 2), strides=None, name='block2_pool')

        self.block3_conv1 = tf.keras.layers.Conv2D(
            256, (3, 3), name='block3_conv1', activation='relu', padding='same')
        self.block3_conv2 = tf.keras.layers.Conv2D(
            256, (3, 3), name='block3_conv2', padding='same')
        self.block3_bn = tf.keras.layers.BatchNormalization()
        self.block3_act = tf.keras.layers.ReLU()
        self.block3_pool = tf.keras.layers.MaxPooling2D(
            (2, 2), strides=None, name='block3_pool')

        self.block4_conv1 = tf.keras.layers.Conv2D(
            512, (3, 3), name='block4_conv1', activation='relu', padding='same')
        self.block4_conv2 = tf.keras.layers.Conv2D(
            512, (3, 3), name='block4_conv2', padding='same')
        self.block4_bn = tf.keras.layers.BatchNormalization()
        self.block4_act = tf.keras.layers.ReLU()
        self.block4_dropout = tf.keras.layers.Dropout(0.5)
        self.block4_pool = tf.keras.layers.MaxPooling2D(
            (2, 2), strides=None, name='block4_pool')

        self.block5_conv1 = tf.keras.layers.Conv2D(
            1024, (3, 3), name='block5_conv1', activation='relu', padding='same')
        self.block5_conv2 = tf.keras.layers.Conv2D(
            1024, (3, 3), name='block5_conv2', padding='same')
        self.block5_bn = tf.keras.layers.BatchNormalization()
        self.block5_act = tf.keras.layers.ReLU()
        self.block5_dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x):
        z1 = self.block1_conv1(x)
        z1 = self.block1_conv2(z1)
        z1 = self.block1_bn(z1)
        z1 = self.block1_act(z1)
        z1_pool = self.block1_pool(z1)

        z2 = self.block2_conv1(z1_pool)
        z2 = self.block2_conv2(z2)
        z2 = self.block2_bn(z2)
        z2 = self.block2_act(z2)
        z2_pool = self.block2_pool(z2)

        z3 = self.block3_conv1(z2_pool)
        z3 = self.block3_conv2(z3)
        z3 = self.block3_bn(z3)
        z3 = self.block3_act(z3)
        z3_pool = self.block3_pool(z3)

        z4 = self.block4_conv1(z3_pool)
        z4 = self.block4_conv2(z4)
        z4 = self.block4_bn(z4)
        z4 = self.block4_act(z4)
        z4_dropout = self.block4_dropout(z4)
        z4_pool = self.block4_pool(z4_dropout)

        z5 = self.block5_conv1(z4_pool)
        z5 = self.block5_conv2(z5)
        z5 = self.block5_bn(z5)
        z5 = self.block5_act(z5)
        z5_dropout = self.block5_dropout(z5)

        return z1, z2, z3, z4_dropout, z5_dropout


class Decoder(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        # Network
        self.block6_up = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.block6_conv1 = tf.keras.layers.Conv2D(
            512, (2, 2), name='block6_conv1', activation='relu', padding='same')
        self.block6_conv2 = tf.keras.layers.Conv2D(
            512, (3, 3), name='block6_conv2', activation='relu', padding='same')
        self.block6_conv3 = tf.keras.layers.Conv2D(
            512, (3, 3), name='block6_conv3', padding='same')
        self.block6_bn = tf.keras.layers.BatchNormalization()
        self.block6_act = tf.keras.layers.ReLU()

        self.block7_up = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.block7_conv1 = tf.keras.layers.Conv2D(
            256, (2, 2), name='block7_conv1', activation='relu', padding='same')
        self.block7_conv2 = tf.keras.layers.Conv2D(
            256, (3, 3), name='block7_conv2', activation='relu', padding='same')
        self.block7_conv3 = tf.keras.layers.Conv2D(
            256, (3, 3), name='block7_conv3', padding='same')
        self.block7_bn = tf.keras.layers.BatchNormalization()
        self.block7_act = tf.keras.layers.ReLU()

        self.block8_up = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.block8_conv1 = tf.keras.layers.Conv2D(
            128, (2, 2), name='block8_conv1', activation='relu', padding='same')
        self.block8_conv2 = tf.keras.layers.Conv2D(
            128, (3, 3), name='block8_conv2', activation='relu', padding='same')
        self.block8_conv3 = tf.keras.layers.Conv2D(
            128, (3, 3), name='block8_conv3', padding='same')
        self.block8_bn = tf.keras.layers.BatchNormalization()
        self.block8_act = tf.keras.layers.ReLU()

        self.block9_up = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.block9_conv1 = tf.keras.layers.Conv2D(
            64, (2, 2), name='block9_conv1', activation='relu', padding='same')
        self.block9_conv2 = tf.keras.layers.Conv2D(
            64, (3, 3), name='block9_conv2', activation='relu', padding='same')
        self.block9_conv3 = tf.keras.layers.Conv2D(
            64, (3, 3), name='block9_conv3', padding='same')
        self.block9_bn = tf.keras.layers.BatchNormalization()
        self.block9_act = tf.keras.layers.ReLU()
        self.output_conv = tf.keras.layers.Conv2D(
            num_classes + 1, (1, 1), name='output_conv', activation='sigmoid')

    def call(self, z1, z2, z3, z4_dropout, z5_dropout):
        z6_up = self.block6_up(z5_dropout)
        z6 = self.block6_conv1(z6_up)
        z6 = tf.keras.layers.concatenate([z4_dropout, z6], axis=3)
        z6 = self.block6_conv2(z6)
        z6 = self.block6_conv3(z6)
        z6 = self.block6_bn(z6)
        z6 = self.block6_act(z6)

        z7_up = self.block7_up(z6)
        z7 = self.block7_conv1(z7_up)
        z7 = tf.keras.layers.concatenate([z3, z7], axis=3)
        z7 = self.block7_conv2(z7)
        z7 = self.block7_conv3(z7)
        z7 = self.block7_bn(z7)
        z7 = self.block7_act(z7)

        z8_up = self.block8_up(z7)
        z8 = self.block8_conv1(z8_up)
        z8 = tf.keras.layers.concatenate([z2, z8], axis=3)
        z8 = self.block8_conv2(z8)
        z8 = self.block8_conv3(z8)
        z8 = self.block8_bn(z8)
        z8 = self.block8_act(z8)

        z9_up = self.block9_up(z8)
        z9 = self.block9_conv1(z9_up)
        z9 = tf.keras.layers.concatenate([z1, z9], axis=3)
        z9 = self.block9_conv2(z9)
        z9 = self.block9_conv3(z9)
        z9 = self.block9_bn(z9)
        z9 = self.block9_act(z9)
        y = self.output_conv(z9)

        return y
