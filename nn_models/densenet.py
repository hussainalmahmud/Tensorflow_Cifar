import tensorflow as tf

class BottleneckLayer(tf.keras.layers.Layer):
    def __init__(self, growth_rate):
        super(BottleneckLayer, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(4 * growth_rate, kernel_size=1, use_bias=False, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(growth_rate, kernel_size=3, use_bias=False, padding='same')

    def call(self, x, training=False):
        output = self.bn1(x, training=training)
        output = tf.keras.activations.relu(output)
        output = self.conv1(output)
        output = self.bn2(output, training=training)
        output = tf.keras.activations.relu(output)
        output = self.conv2(output)
        return tf.concat([x, output], axis=-1)

class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(out_channels, kernel_size=1, use_bias=False, padding='same')
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)

    def call(self, x, training=False):
        output = self.bn(x, training=training)
        output = tf.keras.activations.relu(output)
        output = self.conv(output)
        return self.avgpool(output)

class DenseNet(tf.keras.Model):
    VARIANTS = {
        'densenet121': [6, 12, 24, 16],
        'densenet169': [6, 12, 32, 32],
        'densenet201': [6, 12, 48, 32],
        'densenet264': [6, 12, 64, 48]
    }

    def __init__(self, variant='densenet121', growth_rate=32, num_classes=10):
        super(DenseNet, self).__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported Densenet variant '{variant}'. Supported variants are: {', '.join(self.VARIANTS.keys())}")

        num_blocks = self.VARIANTS[variant]
        num_channels = 2 * growth_rate
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.blocks = []

        for block in num_blocks[:-1]:
            self.blocks.append(self._create_block(growth_rate, num_channels, block))
            num_channels += growth_rate * block
            num_channels = num_channels // 2
            self.blocks.append(TransitionLayer(num_channels))

        self.blocks.append(self._create_block(growth_rate, num_channels, num_blocks[-1]))
        num_channels += growth_rate * num_blocks[-1]
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def _create_block(self, growth_rate, in_channels, num_bottlenecks):
        layers = []
        for _ in range(num_bottlenecks):
            layers.append(BottleneckLayer(growth_rate))
            in_channels += growth_rate
        return tf.keras.Sequential(layers)

    def call(self, x, training=False):
        output = self.conv(x)
        output = self.bn(output, training=training)
        output = tf.keras.activations.relu(output)
        output = self.pool(output)
        for block in self.blocks:
            output = block(output, training=training)
        output = self.avgpool(output)
        return self.fc(output)
 
    def build(self, input_shape=(32, 32, 3)):
        input_tensor = tf.keras.Input(shape=input_shape)
        output_tensor = self.call(input_tensor)
        model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
        return model

# Example usage:
# model_instance = DenseNet(variant='densenet264')
# model = model_instance.build()
# model.summary()




