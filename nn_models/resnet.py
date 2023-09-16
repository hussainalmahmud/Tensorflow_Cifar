import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, MaxPooling2D
from tensorflow.keras.models import Model

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, stride=1, use_bottleneck=False):
        super(ResidualBlock, self).__init__()

        self.use_bottleneck = use_bottleneck
        self.stride = stride
        
        if use_bottleneck:
            self.conv1 = Conv2D(filters // 4, (1, 1), strides=stride, padding="same")
            self.conv2 = Conv2D(filters // 4, kernel_size, strides=1, padding="same")
            self.conv3 = Conv2D(filters, (1, 1), strides=1, padding="same")
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()
            self.bn3 = BatchNormalization()
        else:
            self.conv1 = Conv2D(filters, kernel_size, strides=stride, padding="same")
            self.bn1 = BatchNormalization()
            self.conv2 = Conv2D(filters, kernel_size, strides=1, padding="same")
            self.bn2 = BatchNormalization()
        
        self.shortcut = Conv2D(filters, (1, 1), strides=stride, padding="same") if stride != 1 else lambda x: x

    def call(self, x):
        shortcut = self.shortcut(x)
        if self.use_bottleneck:
            x = self.conv1(x)
            x = self.bn1(x)
            x = ReLU()(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = ReLU()(x)
            x = self.conv3(x)
            x = self.bn3(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = ReLU()(x)
            x = self.conv2(x)
            x = self.bn2(x)
        
        x = Add()([x, shortcut])
        x = ReLU()(x)
        return x

class ResNet:
    def __init__(self, variant='resnet50', num_classes=10):
        configs = {
            'resnet18': [2, 2, 2, 2],
            'resnet34': [3, 4, 6, 3],
            'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
            'resnet152': [3, 8, 36, 3]
        }
        
        self.config = configs[variant]
        self.num_classes = num_classes
        self.use_bottleneck = variant in ['resnet50', 'resnet101', 'resnet152']
        
    def build(self, input_shape=(32, 32, 3)):
        input_tensor = Input(shape=input_shape)
        x = Conv2D(64, (7, 7), padding="same", strides=2)(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((3, 3), strides=2, padding="same")(x)
        
        filters = 64
        for block_config in self.config:
            for i in range(block_config):
                stride = 2 if i == 0 and filters != 64 else 1
                x = ResidualBlock(filters, stride=stride, use_bottleneck=self.use_bottleneck)(x)
            filters *= 2
        
        x = GlobalAveragePooling2D()(x)
        output_tensor = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_tensor, outputs=output_tensor)
        return model

# # Example usage:
# resnet50 = ResNet('resnet152').build()
# resnet50.summary()
