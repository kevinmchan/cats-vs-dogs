from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


BATCH_SIZE = 32
TRAIN_SIZE = 3000
VAL_SIZE = 500
TEST_SIZE = 500
START_FINE_TUNING_LAYER = 144
N_EPOCHS = 5


def model_specification():
    conv_base = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    print("Number of layers in the base model: ", len(conv_base.layers))
    conv_base.trainable = True
    for layer in conv_base.layers[:START_FINE_TUNING_LAYER]:
        layer.trainable =  False
    
    model = models.Sequential(
        layers=[
            conv_base,
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"],
    )
    return model


def image_generators():
    train_dir = "./data/sample/training"
    val_dir = "./data/sample/validation"
    test_dir = "./data/sample/test"

    train_data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    test_data_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_data_gen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=["dog", "cat"],
    )

    val_generator = test_data_gen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=["dog", "cat"],
    )

    test_generator = test_data_gen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=["dog", "cat"],
    )
    return train_generator, val_generator, test_generator


def plot_training_and_validation_performance(history, output):
    import matplotlib.pyplot as plt

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend()

    plt.savefig(output)


def main():
    model = model_specification()
    model.summary()
    train_generator, val_generator, test_generator = image_generators()
    model_checkpoint = ModelCheckpoint(
        "./model/cat_dog_mobilenet_pooled_finetuned_best.h5",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)

    history = model.fit(
        train_generator,
        steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,
        epochs=N_EPOCHS,
        validation_data=val_generator,
        validation_steps=VAL_SIZE // BATCH_SIZE,
        callbacks=[model_checkpoint, es],
    )
    model.load_weights("./model/cat_dog_mobilenet_pooled_finetuned_best.h5")

    plot_training_and_validation_performance(
        history, "./figures/cat_dog_mobilenet_pooled_finetuned_perf.png"
    )
    print(model.evaluate(test_generator, steps=TEST_SIZE // BATCH_SIZE))


if __name__ == "__main__":
    main()
