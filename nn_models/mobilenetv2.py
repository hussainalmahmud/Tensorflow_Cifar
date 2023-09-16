import tensorflow as tf
from tensorflow.keras.layers import Input, DepthwiseConv2D, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def inverted_residual_block(inputs, filters, kernel_size, stride, expansion, block_id):
    x = Conv2D(filters * expansion, (1, 1), strides=(1, 1), padding='same', name=f'expand_{block_id}')(inputs)
    x = BatchNormalization(name=f'expand_BN_{block_id}')(x)
    x = ReLU(name=f'expand_ReLU_{block_id}')(x)

    x = DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=1, padding='same', name=f'depthwise_{block_id}')(x)
    x = BatchNormalization(name=f'depthwise_BN_{block_id}')(x)
    x = ReLU(name=f'depthwise_ReLU_{block_id}')(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name=f'project_{block_id}')(x)
    x = BatchNormalization(name=f'project_BN_{block_id}')(x)
    
    if stride == 1 and inputs.shape[-1] != filters:
        # Pointwise convolution to match the number of channels
        inputs = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name=f'reshape_{block_id}')(inputs)
    
    if stride == 1:
        x = Add(name=f'residual_add_{block_id}')([inputs, x])

    return x


def MobileNetV2(input_shape, num_classes=10):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', name='Conv1')(inputs)
    x = BatchNormalization(name='BN_Conv1')(x)
    x = ReLU(name='ReLU_Conv1')(x)

    x = inverted_residual_block(x, 16, (3, 3), stride=1, expansion=1, block_id=1)

    x = inverted_residual_block(x, 24, (3, 3), stride=2, expansion=6, block_id=2)
    x = inverted_residual_block(x, 24, (3, 3), stride=1, expansion=6, block_id=3)

    x = inverted_residual_block(x, 32, (3, 3), stride=2, expansion=6, block_id=4)
    x = inverted_residual_block(x, 32, (3, 3), stride=1, expansion=6, block_id=5)
    x = inverted_residual_block(x, 32, (3, 3), stride=1, expansion=6, block_id=6)

    x = inverted_residual_block(x, 64, (3, 3), stride=2, expansion=6, block_id=7)
    x = inverted_residual_block(x, 64, (3, 3), stride=1, expansion=6, block_id=8)
    x = inverted_residual_block(x, 64, (3, 3), stride=1, expansion=6, block_id=9)
    x = inverted_residual_block(x, 64, (3, 3), stride=1, expansion=6, block_id=10)

    x = inverted_residual_block(x, 96, (3, 3), stride=1, expansion=6, block_id=11)
    x = inverted_residual_block(x, 96, (3, 3), stride=1, expansion=6, block_id=12)
    x = inverted_residual_block(x, 96, (3, 3), stride=1, expansion=6, block_id=13)

    x = inverted_residual_block(x, 160, (3, 3), stride=2, expansion=6, block_id=14)
    x = inverted_residual_block(x, 160, (3, 3), stride=1, expansion=6, block_id=15)
    x = inverted_residual_block(x, 160, (3, 3), stride=1, expansion=6, block_id=16)

    x = inverted_residual_block(x, 320, (3, 3), stride=1, expansion=6, block_id=17)

    x = Conv2D(1280, (1, 1), strides=(1, 1), padding='same', name='Conv_1')(x)
    x = BatchNormalization(name='BN_Conv_1')(x)
    x = ReLU(name='ReLU_Conv_1')(x)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax', name='Logits')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


