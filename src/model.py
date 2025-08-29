IMG_SIZE = 224  # Standard input size for both models
BATCH_SIZE = 16
NUM_CLASSES = 7  # For HAM10000

model_choice = 'efficientnet'
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam


def focal_loss(alpha=0.25, gamma=2.0):
    def focal(y_true, y_pred):
        epsilon = 1e-9
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = tf.reduce_sum(weight * cross_entropy, axis=1)
        return tf.reduce_mean(loss)
    return focal

# Define input shape
input_shape = (224, 224, 3)

# Base model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = True  # Fine-tune all layers

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(7, activation='softmax')(x)  # 7 classes in HAM10000

# Build model
model = Model(inputs=base_model.input, outputs=outputs)

#  Compile with Focal Loss
model.compile(optimizer=Adam(1e-5), loss=focal_loss(alpha=0.25, gamma=2.0), metrics=['accuracy'])

# Summary
model.summary()
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

y_train_classes = train_generator.classes  # only from training
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_classes),
    y=y_train_classes
)
class_weights = dict(enumerate(class_weights))
aggressive_weights = {cls: weight**1.5 for cls, weight in class_weights.items()}

base_model.trainable = True  # unfreeze if it was frozen

from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss=(focal_loss(alpha=0.25, gamma=2.0)),
    metrics=['accuracy']
)
from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
from sklearn.utils import resample

TARGET_SAMPLES = 1099  # adjust as needed
resampled_dfs = []

for label in df_train['label'].unique():
    df_class = df_train[df_train['label'] == label]

    if len(df_class) < TARGET_SAMPLES:
        df_resampled = resample(df_class, replace=True, n_samples=TARGET_SAMPLES, random_state=42)
    else:
        df_resampled = df_class.sample(TARGET_SAMPLES, random_state=42)

    resampled_dfs.append(df_resampled)

df_train_balanced = pd.concat(resampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
# Train generator from oversampled data
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train_balanced,
    x_col='path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Validation generator from untouched validation data
val_generator = train_datagen.flow_from_dataframe(
    dataframe=df_val,
    x_col='path',
    y_col='label',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Save every epoch
checkpoint_cb = ModelCheckpoint(
    "model_epoch_{epoch:02d}.h5",
    save_freq='epoch',
    verbose=1
)

# Save best model (by val_loss)
best_checkpoint_cb = ModelCheckpoint(
    "best_model.h5",
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

# Early stopping
earlystop_cb = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Optional manual periodic backups
class SaveEveryNEpochs(tf.keras.callbacks.Callback):
    def __init__(self, save_freq=3):
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            self.model.save(f"backup_epoch_{epoch+1}.h5")
            print(f" Backup saved at epoch {epoch+1}")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[checkpoint_cb, best_checkpoint_cb, earlystop_cb, SaveEveryNEpochs(3)]
)
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get true and predicted labels
val_generator.reset()
y_true = val_generator.classes
y_pred_probs = model.predict(val_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# Class names
class_names = list(val_generator.class_indices.keys())

# Report
print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
