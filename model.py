import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the CNN model
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3),
                        activation='relu', 
                        input_shape=(64, 64, 3)
                       ))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), 
                        activation='relu'
                       ))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output for dense layers
model.add(layers.Flatten())

# Dense layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model architecture
model.summary()



# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescale validation set
val_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    'C:/Users/naidu/CodeAlphaTask-1/task-1/GenderDetection/Training',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Load validation data
validation_generator = val_datagen.flow_from_directory(
    'C:/Users/naidu/CodeAlphaTask-1/task-1/GenderDetection/Validation',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)



history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)



# Evaluate the model
test_loss, test_acc = model.evaluate(validation_generator, verbose=2)
print(f'\nTest accuracy: {test_acc}')




# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()




#model.save("cat_and_dog_object_detection.h5")
model.save("Male_Female_Weights.h5")