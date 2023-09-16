import pytest
from nn_models.mobilenetv2 import MobileNetV2
from nn_models.resnet import ResNet
from nn_models.densenet import DenseNet
from nn_models.vgg import VGG16
from main import AVAILABLE_MODELS

def test_available_models():
    expected_models = [
        "MobileNetV2",
        "ResNet18",
        "ResNet34",
        "ResNet50",
        "ResNet101",
        "ResNet152",
        "DenseNet121",
        "VGG16"
    ]

    assert set(AVAILABLE_MODELS) == set(expected_models), f"Expected models: {expected_models}, but got: {AVAILABLE_MODELS}"


def get_model_instance(model_name, num_classes=10):
    """Utility function to get a model instance by name."""
    if model_name == "MobileNetV2":
        return MobileNetV2((32, 32, 3), num_classes)
    elif "ResNet" in model_name:
        return ResNet(model_name.lower(), num_classes).build()
    elif model_name == "DenseNet121":
        return DenseNet("densenet121", num_classes).build()
    elif model_name == "VGG16":
        return VGG16(num_classes).build()
    else:
        raise ValueError("Unknown model name.")

@pytest.mark.parametrize("model_name", ["MobileNetV2", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "DenseNet121", "VGG16"])
def test_input_shape(model_name):
    model_instance = get_model_instance(model_name)
    assert model_instance.input_shape == (None, 32, 32, 3)  # 'None' is for the batch size.
