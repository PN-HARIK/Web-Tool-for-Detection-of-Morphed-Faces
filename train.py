import numpy as np
import matplotlib.pyplot as plt
import os
import random
import itertools
from PIL import Image, ImageChops, ImageEnhance

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


np.random.seed(2)

# ---------------------------
# ELA Conversion Function
# ---------------------------
def convert_to_ela_image(path, quality):
    temp_filename = "temp.jpg"

    image = Image.open(path).convert("RGB")
    image.save(temp_filename, "JPEG", quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])

    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image


# ---------------------------
# Image Preparation
# ---------------------------
image_size = (128, 128)

def prepare_image(image_path):
    return np.array(
        convert_to_ela_image(image_path, 90).resize(image_size)
    ).flatten() / 255.0


# ---------------------------
# Dataset Preparation
# ---------------------------
X = []
Y = []

# Real Images
real_path = r"G:\Main\Data\training_real"

for dirname, _, filenames in os.walk(real_path):
    for filename in filenames:
        if filename.endswith(("jpg", "png")):
            full_path = os.path.join(dirname, filename)
            X.append(prepare_image(full_path))
            Y.append(1)

random.shuffle(X)

X = X[:2100]
Y = Y[:2100]


# Fake Images
fake_path = r"G:\Main\Data\training_fake"

for dirname, _, filenames in os.walk(fake_path):
    for filename in filenames:
        if filename.endswith(("jpg", "png")):
            full_path = os.path.join(dirname, filename)
            X.append(prepare_image(full_path))
            Y.append(0)


X = np.array(X)
Y = to_categorical(Y, 2)

X = X.reshape(-1, 128, 128, 3)

# ---------------------------
# Train Test Split
# ---------------------------
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=5
)


# ---------------------------
# CNN Model
# ---------------------------
def build_model():

    model = Sequential()

    model.add(
        Conv2D(
            filters=32,
            kernel_size=(5, 5),
            activation="relu",
            input_shape=(128, 128, 3),
        )
    )

    model.add(Conv2D(32, (5, 5), activation="relu"))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (5, 5), activation="relu"))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (5, 5), activation="relu"))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation="softmax"))

    return model


model = build_model()

# ---------------------------
# Compile Model
# ---------------------------
epochs = 30
batch_size = 32

optimizer = Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------
# Early Stopping
# ---------------------------
early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    mode="max"
)

# ---------------------------
# Train Model
# ---------------------------
history = model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, Y_val),
    callbacks=[early_stopping],
)

# ---------------------------
# Save Model
# ---------------------------
model.save("Pretrainedmodel.h5")


# ---------------------------
# Plot Accuracy and Loss
# ---------------------------
fig, ax = plt.subplots(2, 1)

ax[0].plot(history.history["loss"], label="Training Loss")
ax[0].plot(history.history["val_loss"], label="Validation Loss")
ax[0].legend()

ax[1].plot(history.history["accuracy"], label="Training Accuracy")
ax[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
ax[1].legend()

plt.show()


# ---------------------------
# Confusion Matrix
# ---------------------------
def plot_confusion_matrix(cm, classes, cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()


Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_val, axis=1)

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

plot_confusion_matrix(confusion_mtx, classes=["Fake", "Real"])
plt.show()


# ---------------------------
# Single Image Prediction
# ---------------------------
class_names = ["fake", "real"]

test_image = r"test.jpg"

image = prepare_image(test_image)
image = image.reshape(-1, 128, 128, 3)

prediction = model.predict(image)

predicted_class = np.argmax(prediction, axis=1)[0]

print(
    f"Class: {class_names[predicted_class]} | Confidence: {np.max(prediction)*100:.2f}%"
)