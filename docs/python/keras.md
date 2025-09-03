# Keras

## Installation
```bash
# Keras 3 (multi-backend)
pip install --upgrade keras

# With specific backend dependencies
pip install --upgrade keras-cv keras-hub keras  # For computer vision and NLP
pip install --upgrade jax[cpu]    # For JAX backend
pip install --upgrade tensorflow  # For TensorFlow backend  
pip install --upgrade torch       # For PyTorch backend

# Check version
python -c "import keras; print(keras.__version__)"
```

## Backend Configuration
```python
# Set backend before importing Keras (method 1)
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # or "jax", "torch"

# Or via environment variable (method 2)
export KERAS_BACKEND="jax"

# Or in Colab/Jupyter
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras

# Check current backend
print(keras.config.backend())
```

## Import Essentials
```python
import keras
from keras import layers, models, utils, optimizers, losses, metrics
from keras import ops  # Backend-agnostic operations
import numpy as np
import matplotlib.pyplot as plt
```

## Models

### Sequential Model
```python
# Method 1: List of layers
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Method 2: Add layers incrementally
model = keras.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(784,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))
```

### Functional API
```python
inputs = layers.Input(shape=(784,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Multi-input/output example
input1 = layers.Input(shape=(10,), name='input1')
input2 = layers.Input(shape=(5,), name='input2')

x1 = layers.Dense(64, activation='relu')(input1)
x2 = layers.Dense(32, activation='relu')(input2)

concatenated = layers.Concatenate()([x1, x2])
output1 = layers.Dense(1, activation='sigmoid', name='output1')(concatenated)
output2 = layers.Dense(3, activation='softmax', name='output2')(concatenated)

model = keras.Model(inputs=[input1, input2], outputs=[output1, output2])
```

### Model Subclassing
```python
class MyModel(keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)

    def get_config(self):
        return {"num_classes": self.dense2.units}

model = MyModel(num_classes=10)
```

## Layers

### Dense (Fully Connected)
```python
layers.Dense(64, activation='relu')
layers.Dense(10, activation='softmax', use_bias=False)
layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.01))
```

### Convolutional Layers
```python
# 2D Convolution
layers.Conv2D(32, (3, 3), activation='relu', padding='same')
layers.Conv2D(64, 3, strides=2, activation='relu')

# Depthwise Separable
layers.SeparableConv2D(64, (3, 3), activation='relu')

# Transposed Convolution (Deconvolution)
layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same')

# 1D and 3D variants
layers.Conv1D(32, 3, activation='relu')
layers.Conv3D(32, (3, 3, 3), activation='relu')
```

### Pooling Layers
```python
layers.MaxPooling2D((2, 2))
layers.AveragePooling2D((2, 2), strides=2)
layers.GlobalMaxPooling2D()
layers.GlobalAveragePooling2D()
```

### Recurrent Layers
```python
layers.LSTM(64, return_sequences=True)
layers.GRU(32, dropout=0.2, recurrent_dropout=0.2)
layers.SimpleRNN(32)

# Bidirectional
layers.Bidirectional(layers.LSTM(64))

# Stacked RNNs
layers.LSTM(64, return_sequences=True)
layers.LSTM(32)
```

### Attention and Transformer Layers
```python
# Multi-head attention
layers.MultiHeadAttention(num_heads=8, key_dim=64)

# Self-attention
layers.Attention()

# Add & Normalize
layers.LayerNormalization()
layers.Add()
```

### Normalization Layers
```python
layers.BatchNormalization()
layers.LayerNormalization()
layers.GroupNormalization(groups=32)
layers.UnitNormalization()
```

### Regularization
```python
layers.Dropout(0.2)
layers.AlphaDropout(0.1)  # For SELU activation
layers.SpatialDropout2D(0.2)  # For convolutional layers
layers.GaussianDropout(0.1)
layers.GaussianNoise(0.1)
```

### Activation Layers
```python
layers.ReLU()
layers.LeakyReLU(negative_slope=0.3)
layers.ELU(alpha=1.0)
layers.Softmax(axis=-1)
layers.Activation('tanh')

# Advanced activations
layers.PReLU()
layers.ThresholdedReLU(theta=1.0)
layers.Softplus()
```

### Utility Layers
```python
layers.Flatten()
layers.Reshape((3, 4))
layers.Permute((2, 1))
layers.RepeatVector(3)
layers.Lambda(lambda x: x ** 2)
layers.Cropping2D(cropping=((1, 1), (2, 2)))
layers.ZeroPadding2D(padding=(1, 1))
```

## Model Compilation

### Basic Compilation
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Custom Configuration
```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(),
        keras.metrics.TopKCategoricalAccuracy(k=5)
    ]
)
```

### Multiple Losses (Multi-output)
```python
model.compile(
    optimizer='adam',
    loss={
        'output1': 'binary_crossentropy',
        'output2': 'categorical_crossentropy'
    },
    loss_weights={'output1': 1.0, 'output2': 0.2},
    metrics={
        'output1': ['accuracy'],
        'output2': ['accuracy', 'top_k_categorical_accuracy']
    }
)
```

## Optimizers
```python
# SGD with momentum
keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Adam variants
keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
keras.optimizers.Adamax(learning_rate=0.002)

# Other optimizers
keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
keras.optimizers.Adagrad(learning_rate=0.01)
keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
```

### Learning Rate Schedules
```python
# Exponential decay
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    decay_rate=0.9
)

# Cosine decay
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.1,
    decay_steps=1000
)

# Piecewise constant
lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[100000, 110000],
    values=[1.0, 0.5, 0.1]
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

## Loss Functions

### Classification
```python
keras.losses.BinaryCrossentropy()
keras.losses.BinaryFocalCrossentropy(alpha=0.25, gamma=2.0)
keras.losses.CategoricalCrossentropy()
keras.losses.SparseCategoricalCrossentropy()
keras.losses.KLDivergence()
```

### Regression
```python
keras.losses.MeanSquaredError()
keras.losses.MeanAbsoluteError()
keras.losses.MeanAbsolutePercentageError()
keras.losses.MeanSquaredLogarithmicError()
keras.losses.Huber(delta=1.0)
```

### Custom Loss
```python
def custom_loss(y_true, y_pred):
    return ops.mean(ops.square(y_true - y_pred))

# Or as a class
class CustomLoss(keras.losses.Loss):
    def __init__(self, name="custom_loss"):
        super().__init__(name=name)
    
    def call(self, y_true, y_pred):
        return ops.mean(ops.square(y_true - y_pred))

model.compile(optimizer='adam', loss=CustomLoss())
```

## Training

### Basic Training
```python
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_val, y_val),
    verbose=1
)
```

### Advanced Training with Callbacks
```python
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=0.001
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        write_graph=True
    )
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    callbacks=callbacks,
    verbose=1
)
```

### Custom Training Step
```python
class CustomModel(keras.Model):
    def train_step(self, data):
        x, y = data
        
        with keras.backends.gradienttape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
```

## Data Processing

### Image Data Loading and Augmentation
```python
# Create a dataset from a directory of images
train_ds = keras.utils.image_dataset_from_directory(
    directory='train_directory',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224)
)

# Define data augmentation layers
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Apply augmentation to the dataset
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

# Prefetch for performance
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Fit with the dataset
model.fit(train_ds, epochs=10, validation_data=val_ds)
```

### tf.data Integration
```python
# Using tf.data with Keras
import tensorflow as tf

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

model.fit(train_ds, epochs=10)
```

## Evaluation and Prediction

### Model Evaluation
```python
# Evaluate on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Detailed evaluation with multiple metrics
results = model.evaluate(
    x_test, y_test,
    batch_size=32,
    return_dict=True,
    verbose=1
)
```

### Predictions
```python
# Predictions
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

# Single prediction
single_pred = model.predict(np.expand_dims(single_sample, axis=0))

# Prediction with custom batch size
predictions = model.predict(x_test, batch_size=64, verbose=1)
```

## Model Persistence

### Save/Load Model
```python
# Save entire model
model.save('my_model.keras')  # Recommended format
model.save('my_model.h5')     # Legacy format

# Load model
loaded_model = keras.models.load_model('my_model.keras')

# Save/load weights only
model.save_weights('weights.weights.h5')
model.load_weights('weights.weights.h5')
```

### Export to Different Formats
```python
# SavedModel format (for TensorFlow Serving)
model.export('saved_model_dir')

# Load exported model
imported_model = keras.models.load_model('saved_model_dir')
```

## Callbacks

### Built-in Callbacks
```python
# Early stopping
keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Learning rate scheduling
keras.callbacks.LearningRateScheduler(
    lambda epoch: 0.1 * 0.95 ** epoch
)

# CSV logger
keras.callbacks.CSVLogger('training.log')

# Model checkpointing
keras.callbacks.ModelCheckpoint(
    'model_{epoch:02d}_{val_loss:.2f}.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)
```

### Custom Callbacks
```python
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")
    
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') > 0.95:
            print("Reached 95% accuracy, stopping training!")
            self.model.stop_training = True
    
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            print(f"Finished batch {batch}")

model.fit(x_train, y_train, callbacks=[CustomCallback()])
```

## Metrics

### Built-in Metrics
```python
# Classification metrics
keras.metrics.Accuracy()
keras.metrics.BinaryAccuracy()
keras.metrics.CategoricalAccuracy()
keras.metrics.SparseCategoricalAccuracy()
keras.metrics.TopKCategoricalAccuracy(k=5)
keras.metrics.Precision()
keras.metrics.Recall()
keras.metrics.F1Score()
keras.metrics.AUC()

# Regression metrics
keras.metrics.MeanSquaredError()
keras.metrics.MeanAbsoluteError()
keras.metrics.RootMeanSquaredError()
keras.metrics.MeanAbsolutePercentageError()
```

### Custom Metrics
```python
class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall = keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + keras.backend.epsilon()))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
```

## Transfer Learning

### Using Pre-trained Models
```python
# Load pre-trained model without top layers
base_model = keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom head
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Fine-tuning: unfreeze some layers
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False
```

## Advanced Features

### Mixed Precision Training
```python
# Enable mixed precision
keras.mixed_precision.set_global_policy('mixed_float16')

# Custom model with loss scaling
class MixedPrecisionModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
    
    def train_step(self, data):
        x, y = data
        
        with keras.backend.gradienttape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
            # Scale loss for mixed precision
            scaled_loss = self.optimizer.get_scaled_loss(loss)
        
        scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
```

### Custom Layers
```python
class CustomDense(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name='kernel'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        super().build(input_shape)
    
    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
```

### Model Visualization
```python
# Plot model architecture
keras.utils.plot_model(
    model,
    to_file='model.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',
    dpi=96
)

# Model summary
model.summary()

# Layer-wise summary
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")
    print(f"  Input shape: {layer.input_shape}")
    print(f"  Output shape: {layer.output_shape}")
    print(f"  Parameters: {layer.count_params()}")
```

## Hyperparameter Tuning with KerasTuner
```python
# Install: pip install keras-tuner

import keras_tuner

def build_model(hp):
    model = keras.Sequential([
        layers.Dense(
            hp.Int('units_1', 32, 512, step=32),
            activation='relu'
        ),
        layers.Dropout(hp.Float('dropout_1', 0, 0.5, step=0.1)),
        layers.Dense(
            hp.Int('units_2', 32, 512, step=32),
            activation='relu'
        ),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20
)

tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
best_model = tuner.get_best_models()[0]
```

## KerasHub and KerasCV

### KerasHub (NLP)
```python
# Install: pip install keras-hub

import keras_hub

# Text classification with BERT
classifier = keras_hub.models.BertClassifier.from_preset(
    "bert_base_en_uncased",
    num_classes=2
)

# Text generation with GPT
generator = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")
output = generator.generate("The weather today", max_length=50)
```

### KerasCV (Computer Vision)
```python
# Install: pip install keras-cv

import keras_cv

# Data augmentation
augment = keras_cv.layers.RandAugment(
    value_range=(0, 255),
    magnitude=0.5,
    augmentations_per_image=3
)

# Object detection
detector = keras_cv.models.RetinaNet.from_preset(
    "retinanet_resnet50_pascalvoc"
)
```

## Debugging and Performance

### Debugging
```python
# Enable eager execution for debugging
# tf.config.run_functions_eagerly(True)  # For TensorFlow backend

# Add debug prints in custom layers/models
class DebugLayer(layers.Layer):
    def call(self, inputs):
        keras.utils.print_msg(f"Input shape: {ops.shape(inputs)}")
        return inputs

# Check for NaN values
def check_nan(tensor, name):
    if ops.any(ops.isnan(tensor)):
        print(f"NaN detected in {name}")
    return tensor
```

### Performance Optimization
```python
# Use mixed precision
keras.mixed_precision.set_global_policy('mixed_float16')

# Enable XLA compilation (TensorFlow backend)
# model.compile(optimizer='adam', loss='mse', jit_compile=True)

# Profile training
# keras.utils.Progbar for custom progress bars
# Use TensorBoard for performance profiling

# Memory optimization
keras.backend.clear_session()  # Clear session to free memory
```

## Best Practices

1. **Model Design**: Start simple, add complexity gradually
2. **Data**: Normalize inputs, use data augmentation for images
3. **Training**: Use callbacks for early stopping and learning rate scheduling
4. **Validation**: Always use validation data to monitor overfitting
5. **Reproducibility**: Set random seeds and use `keras.utils.set_random_seed()`
6. **Save Models**: Use `.keras` format for better compatibility
7. **Backend Choice**: JAX for research, TensorFlow for production, PyTorch for flexibility
8. **Memory Management**: Use `keras.backend.clear_session()` to free memory
9. **Debugging**: Use model summaries and visualization tools
10. **Performance**: Use mixed precision and appropriate batch sizes