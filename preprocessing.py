import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to dataset
data_dir = "data/train"

def load_data(data_dir="data/", img_size=(128,128), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train = datagen.flow_from_directory(
        data_dir, target_size=img_size, subset="training", batch_size=batch_size
    )
    
    val = datagen.flow_from_directory(
        data_dir, target_size=img_size, subset="validation", batch_size=batch_size
    )
    
    return train, val
