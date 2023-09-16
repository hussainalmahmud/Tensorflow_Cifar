import os
import click
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import read_data, data_augmentation_generator
from nn_models.mobilenetv2 import MobileNetV2
from nn_models.resnet import ResNet
from nn_models.densenet import DenseNet
from nn_models.vgg import VGG16


@click.group()
def cli():
    """
    A command line interface for the CIFAR10 and CIFAR100 datasets.
    You can train models and test them using this tool.

    Usage:

    - Training:
    To train a model, specify the model type, dataset, epochs, and batch size.
    For example, to train the MobileNetV2 model on CIFAR-10 for 5 epochs with a batch size of 128:
        python main.py train --dataset=cifar100 --epochs=5 --batch_size=128 --model=MobileNetV2

    - Testing:
    To test a model, specify the model and dataset.
    For example, to test the MobileNetV2 model on CIFAR-10:
        python main.py test --dataset=cifar10 --model=MobileNetV2

    For more details on any command:
        python main.py COMMAND --help
    """
    pass  # pylint: disable=unnecessary-pass


def list_models(ctx, _, value):
    if value:
        click.echo("Available models are:")
        for model in AVAILABLE_MODELS:
            click.echo(f"- {model}")
        ctx.exit()


AVAILABLE_MODELS = [
    "MobileNetV2",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "DenseNet121",
    "VGG16",
]  # Add or remove model names as needed.


@cli.command()
@click.option(
    "--dataset",
    default="cifar10",
    help="Dataset to train on. Either cifar10 or cifar100",
)
@click.option("--epochs", default=1, help="Number of epochs to train for")
@click.option("--batch_size", default=64, help="Batch size")
@click.option(
    "--list_models",
    is_flag=True,
    callback=list_models,
    expose_value=False,
    is_eager=True,
    help="List available models",
)
@click.option(
    "--model", default="MobileNetV2", help="Model to train. E.g., MobileNetV2"
)
def train(model, dataset, epochs, batch_size):
    """
    Train a specified model on a chosen dataset.

    For instance, to train the MobileNetV2 model on the CIFAR-10 dataset for 5 epochs with a batch size of 128, run:
    python main.py train --dataset=cifar100 --epochs=5 --batch_size=128 --model=MobileNetV2
    """
    datagen = data_augmentation_generator()
    print("Loading data...")
    print("Dataset: ", dataset)
    print("Model Name: ", model)

    x_train, y_train, x_val, y_val, _, _ = read_data(dataset)

    if dataset == "cifar10":
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100
    else:
        raise ValueError("dataset must be either 'cifar10' or 'cifar100'")

    if model == "MobileNetV2":
        model_instance = MobileNetV2((32, 32, 3), num_classes)
    elif model == "ResNet18":
        model_instance = ResNet("resnet18", num_classes).build()
    elif model == "ResNet34":
        model_instance = ResNet("resnet34", num_classes).build()
    elif model == "ResNet50":
        model_instance = ResNet("resnet50", num_classes).build()
    elif model == "ResNet101":
        model_instance = ResNet("resnet101", num_classes).build()
    elif model == "ResNet152":
        model_instance = ResNet("resnet152", num_classes).build()
    elif model == "DenseNet121":
        model_instance = DenseNet("densenet121", num_classes).build()
    elif model == "VGG16":
        model_instance = VGG16(num_classes).build()
    else:
        raise ValueError("model must be selected e.g., 'MobileNetV2', 'ResNet18'")

    model_instance.summary()
    model_instance.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    hist = model_instance.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_val, y_val),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ],
    )
    if not os.path.exists("model"):
        os.makedirs("model")

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(f"model_output/hist_{model}_{dataset}.csv", encoding="utf-8", index=False)
    if not os.path.exists(f"model_output/output_{model}_{dataset}"):
        os.makedirs(f"model_output/output_{model}_{dataset}")
    model_instance.save(f"model_output/output_{model}_{dataset}")

    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")
    plt.savefig(f"model_output/output_{model}_{dataset}/accuracy.png")
    plt.clf()


@cli.command()
@click.option("--model", default="MobileNetV2", help="Model to test. E.g., MobileNetV2")
@click.option(
    "--dataset",
    default="cifar10",
    help="Dataset to test on. Either cifar10 or cifar100",
)
def test(model, dataset):
    """Test the model"""
    print("Loading data...")
    print("Dataset: ", dataset)
    print("Model Name: ", model)
    model_instance = tf.keras.models.load_model(
        f"model_output/output_{model}_{dataset}"
    )
    _, _, _, _, x_test, y_test = read_data(dataset)
    loss, acc = model_instance.evaluate(x_test, y_test)
    print("Test accuracy: ", acc)
    print("Test loss: ", loss)


if __name__ == "__main__":
    cli()
