# TensorFlow

## Installation
```bash
# CPU only
pip install tensorflow

# GPU support (requires CUDA)
pip install tensorflow[and-cuda]

# Development version
pip install tf-nightly

# Specific version
pip install tensorflow==2.15.0

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Import Essentials
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
```

## Tensor Basics

### Creating Tensors
```python
# From Python lists
tensor = tf.constant([1, 2, 3, 4])
matrix = tf.constant([[1, 2], [3, 4]])

# Zeros and ones
zeros = tf.zeros([3, 4])
ones = tf.ones([2, 3])
identity = tf.eye(3)

# Random tensors
random_normal = tf.random.normal([3, 3])
random_uniform = tf.random.uniform([2, 2], minval=0, maxval=1)

# From numpy
numpy_array = np.array([1, 2, 3])
tensor_from_numpy = tf.constant(numpy_array)

# Range tensors
range_tensor = tf.range(10)
linspace = tf.linspace(0.0, 1.0, 5)
```

### Tensor Properties
```python
tensor = tf.random.normal([3, 4, 5])

print(tensor.shape)        # TensorShape([3, 4, 5])
print(tensor.dtype)        # tf.float32
print(tensor.numpy())      # convert to numpy array
print(tf.rank(tensor))     # number of dimensions (3)
print(tf.size(tensor))     # total number of elements (60)
```

### Tensor Operations
```python
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

# Arithmetic operations
c = a + b                  # tf.add(a, b)
c = a - b                  # tf.subtract(a, b)
c = a * b                  # element-wise multiplication
c = a / b                  # element-wise division
c = tf.matmul(a, b)        # matrix multiplication
c = a @ b                  # alternative matrix multiplication

# Reductions
tf.reduce_sum(tensor)      # sum all elements
tf.reduce_mean(tensor)     # mean of all elements
tf.reduce_max(tensor)      # maximum element
tf.reduce_min(tensor)      # minimum element
tf.reduce_sum(tensor, axis=1)  # sum along axis 1
```

### Reshaping and Indexing
```python
tensor = tf.random.normal([12])

# Reshaping
reshaped = tf.reshape(tensor, [3, 4])
expanded = tf.expand_dims(tensor, axis=0)  # add dimension
squeezed = tf.squeeze(expanded)            # remove dimension

# Indexing and slicing
tensor[0]          # first element
tensor[1:4]        # slice
tensor[:, 1]       # all rows, column 1
tensor[..., -1]    # last element along last axis

# Advanced indexing
tf.gather(tensor, indices=[0, 2, 4])  # gather specific indices
tf.boolean_mask(tensor, tensor > 0)   # boolean masking
```

## Variables and GradientTape

### Variables
```python
# Creating variables
var = tf.Variable(3.0, name="my_variable")
matrix_var = tf.Variable([[1.0, 2.0], [3.0, 4.0]])

# Updating variables
var.assign(5.0)
var.assign_add(2.0)  # add 2 to current value
var.assign_sub(1.0)  # subtract 1 from current value
```

### Automatic Differentiation
```python
# Using GradientTape for automatic differentiation
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x**2 + 2*x + 1

# Compute gradient
dy_dx = tape.gradient(y, x)  # dy/dx = 2*x + 2 = 8

# Multiple variables
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x1**2 + x2**2

gradients = tape.gradient(y, [x1, x2])
```

## Neural Networks with Keras

### Sequential Model
```python
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Alternative syntax
model = tf.keras.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(784,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))
```

### Functional API
```python
inputs = layers.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

### Custom Model (Subclassing)
```python
class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)

model = MyModel()
```

## Common Layers

### Dense (Fully Connected)
```python
layers.Dense(64, activation='relu')
layers.Dense(10, activation='softmax', use_bias=False)
```

### Convolutional Layers
```python
layers.Conv1D(32, 3, activation='relu')
layers.Conv2D(32, (3, 3), activation='relu', padding='same')
layers.Conv3D(16, (3, 3, 3), activation='relu')

# Separable convolutions
layers.SeparableConv2D(64, (3, 3), activation='relu')
layers.DepthwiseConv2D((3, 3), activation='relu')
```

### Pooling Layers
```python
layers.MaxPooling2D((2, 2))
layers.AveragePooling2D((2, 2))
layers.GlobalMaxPooling2D()
layers.GlobalAveragePooling2D()
```

### Recurrent Layers
```python
layers.SimpleRNN(32)
layers.LSTM(64, return_sequences=True)
layers.GRU(32, dropout=0.2, recurrent_dropout=0.2)
layers.Bidirectional(layers.LSTM(32))
```

### Normalization and Regularization
```python
layers.BatchNormalization()
layers.LayerNormalization()
layers.Dropout(0.5)
layers.AlphaDropout(0.1)  # for SELU activation
```

### Activation Layers
```python
layers.ReLU()
layers.LeakyReLU(alpha=0.3)
layers.ELU(alpha=1.0)
layers.Softmax()
layers.Activation('tanh')
```

## Model Compilation and Training

### Compile Model
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Custom optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
```

### Training
```python
# Basic training
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_val, y_val),
    verbose=1
)

# With callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    callbacks=callbacks
)
```

## Data Loading and Preprocessing

### tf.data API
```python
# From arrays
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.batch(32).shuffle(1000).prefetch(tf.data.AUTOTUNE)

# From files
dataset = tf.data.Dataset.list_files("path/to/images/*.jpg")
dataset = dataset.map(preprocess_function)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Image data from directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    'path/to/train',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)
```

### Data Augmentation
```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Apply to model
model = tf.keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(32, 3, activation='relu'),
    # ... rest of model
])
```

## Loss Functions and Optimizers

### Common Loss Functions
```python
# Classification
tf.keras.losses.BinaryCrossentropy()
tf.keras.losses.CategoricalCrossentropy()
tf.keras.losses.SparseCategoricalCrossentropy()

# Regression
tf.keras.losses.MeanSquaredError()
tf.keras.losses.MeanAbsoluteError()
tf.keras.losses.Huber()

# Custom loss
@tf.function
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
```

### Optimizers
```python
tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
tf.keras.optimizers.RMSprop(learning_rate=0.001)
tf.keras.optimizers.Adagrad(learning_rate=0.01)

# Learning rate schedules
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True
)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
```

## Model Evaluation and Prediction

### Evaluation
```python
# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.4f}")

# Custom metrics
test_loss = tf.keras.metrics.Mean()
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    
    test_loss(t_loss)
    test_accuracy(labels, predictions)
```

### Predictions
```python
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

# Prediction on single sample
single_prediction = model.predict(tf.expand_dims(single_image, 0))
```

## Model Saving and Loading

### Save/Load Entire Model
```python
# Save model
model.save('my_model.h5')  # HDF5 format
model.save('my_model')     # SavedModel format

# Load model
loaded_model = tf.keras.models.load_model('my_model.h5')
loaded_model = tf.keras.models.load_model('my_model')
```

### Save/Load Weights Only
```python
# Save weights
model.save_weights('model_weights.h5')

# Load weights
model.load_weights('model_weights.h5')
```

### Checkpoints
```python
checkpoint_path = "training_checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)

# Load latest checkpoint
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
```

## Custom Training Loops

### Basic Custom Training
```python
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)

# Training loop
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    for images, labels in train_ds:
        train_step(images, labels)
    
    print(f'Epoch {epoch + 1}, '
          f'Loss: {train_loss.result():.4f}, '
          f'Accuracy: {train_accuracy.result():.4f}')
```

## Transfer Learning

### Using Pre-trained Models
```python
# Load pre-trained model
base_model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom head
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Fine-tuning: unfreeze some layers
base_model.trainable = True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
```

## TensorBoard Integration

### Setting up TensorBoard
```python
# TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_val, y_val),
    callbacks=[tensorboard_callback]
)

# Custom scalars
with tf.summary.create_file_writer(log_dir).as_default():
    tf.summary.scalar('learning_rate', lr, step=epoch)
    tf.summary.scalar('accuracy', acc, step=epoch)
```

## TensorFlow Lite (Mobile/Edge Deployment)

### Convert to TensorFlow Lite
```python
# Convert Keras model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Quantization for smaller model size
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()
```

### Run TensorFlow Lite Model
```python
# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model
input_data = np.array(test_image, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
```

## Advanced Features

### Mixed Precision Training
```python
# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Loss scaling for numerical stability
optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

# In custom training loop
with tf.GradientTape() as tape:
    predictions = model(x, training=True)
    loss = compute_loss(y, predictions)
    scaled_loss = optimizer.get_scaled_loss(loss)

scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
gradients = optimizer.get_unscaled_gradients(scaled_gradients)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

### Distributed Training
```python
# Multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Multi-worker strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    multi_worker_dataset = strategy.distribute_datasets_from_function(
        dataset_fn
    )
    multi_worker_model = create_model()
```

### Model Optimization
```python
# Pruning
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50,
        final_sparsity=0.80,
        begin_step=0,
        end_step=end_step
    )
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# Quantization-aware training
quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)
```

## Debugging and Performance

### Profiling
```python
# Profile with TensorBoard
tf.profiler.experimental.start(log_dir)
# ... training code ...
tf.profiler.experimental.stop()

# Profile specific functions
with tf.profiler.experimental.Trace('train', step_num=step):
    train_step()
```

### Memory Management
```python
# Limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set memory limit
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
)
```

## Common Patterns and Best Practices

- Use `tf.function` for performance optimization
- Enable mixed precision for faster training on modern GPUs  
- Use `tf.data` for efficient data loading and preprocessing
- Implement proper validation and early stopping
- Save model checkpoints regularly
- Use TensorBoard for monitoring training progress
- Apply data augmentation to improve generalization
- Consider transfer learning for faster convergence
- Use appropriate batch sizes based on available memory
- Normalize input data for better training stability