import os
import pandas as pd
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import matplotlib as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# --------tiền xử lí--------
# Hàm giải nén và tạo label
def extract_paths_and_labels(folder_path):
    paths_and_labels = []
    for label in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label)
        for image_file in os.listdir(label_folder):
            image_path = os.path.join(label_folder, image_file)
            paths_and_labels.append((image_path, label))
    return paths_and_labels


def lr_schedule(epoch):
    initial_lr = 0.001
    decay_factor = 0.5
    decay_epochs = 10
    return initial_lr * (decay_factor ** (epoch // decay_epochs))


# định nghĩa 1 số hằng số
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.005

data_root = "D:\Test_Cut_Frame\DATA"
train_folder = os.path.join(data_root, "DATA_TRAINING")
test_folder = os.path.join(data_root, "DATA_TEST")

train_labels_and_paths = extract_paths_and_labels(train_folder)
print(train_labels_and_paths.__sizeof__())
val_labels_and_paths = extract_paths_and_labels(test_folder)
print(val_labels_and_paths.__sizeof__())
columns = ['image_path', 'label']

df = pd.DataFrame(train_labels_and_paths, columns=columns)
train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
test_df = pd.DataFrame(train_labels_and_paths, columns=columns)

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_dataset = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,

)

valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
valid_dataset = valid_datagen.flow_from_dataframe(
    valid_df,
    x_col='image_path',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_dataset = valid_datagen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

# Xây dựng model
model = keras.Sequential([
    keras.Input(shape=(*IMAGE_SIZE, 3)),

    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPool2D(),
    layers.Dropout(0.25),

    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPool2D(),
    layers.Dropout(0.25),

    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPool2D(),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dropout(0.25),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.BatchNormalization(),

    layers.Dense(16, activation="relu"),
    layers.Dropout(0.25),
    layers.BatchNormalization(),

    # output gồm 4 đầu ra tương ứng với 4 loại bệnh
    layers.Dense(4, activation="softmax"),

])

print(model.summary())

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.005,decay_rate=0.7,decay_steps=10000), epsilon=1e-6),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],

)

callbacks = [

    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="loss"),
    keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=2, min_lr=4e-4, mode="min"),

]
model.fit(
    train_dataset,
    batch_size=BATCH_SIZE,
    validation_data=valid_dataset,
    callbacks=callbacks,
    epochs=EPOCHS,
    verbose=1,
    initial_epoch=1
)

