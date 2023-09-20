import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.models import Model

class VGGBlock(tf.keras.layers.Layer):
    def __init__(self, filters, repetitions, kernel_size=3):
        super(VGGBlock, self).__init__()
        self.filters = filters
        self.repetitions = repetitions
        self.kernel_size = kernel_size
        self.convs = [Conv2D(filters, kernel_size, padding='same', activation='relu') for _ in range(repetitions)]
        
    def call(self, x):
        for i in range(self.repetitions):
            x = self.convs[i](x)
        return MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

class VGG16:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        
    def build(self, input_shape=(32, 32, 3)):
        input_tensor = Input(shape=input_shape)
        
        # Define VGG16 architecture
        x = VGGBlock(64, 2)(input_tensor)
        x = VGGBlock(128, 2)(x)
        x = VGGBlock(256, 3)(x)
        x = VGGBlock(512, 3)(x)
        x = VGGBlock(512, 3)(x)
        
        # Fully connected layers
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)
        output_tensor = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=input_tensor, outputs=output_tensor)
        return model

# Example usage:
# vgg16 = VGG16(num_classes=10).build()
# vgg16.summary()
