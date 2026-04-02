import tensorflow as tf
from tensorflow import keras
import argparse
import os

def retrain_model():
    """Fine-tunes the deployed model on newly ingested match footage."""
    print("Initiating automated MLOps retraining sequence...")
    
    model_path = 'models/final_model.h5'
    new_data_dir = "data/new_training_data/"
    
    # Check if there is new data
    if not os.path.exists(new_data_dir) or len(os.listdir(new_data_dir)) == 0:
        print("No new data found for retraining. Aborting.")
        return

    # 1. Load the active production model
    print("Loading active tactical model...")
    model = tf.keras.models.load_model(model_path)
    
    # 2. Prepare the new data pipeline
    print("Preparing new data pipeline...")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        new_data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(224, 224),
        batch_size=32
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        new_data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(224, 224),
        batch_size=32
    )

    # Normalization layer to ensure the model focuses on tactical patterns, not lighting
    normalization_layer = keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # 3. Compile with an ultra-low learning rate to prevent catastrophic forgetting
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 4. Fine-tune on the new data
    print("Beginning fine-tuning epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        verbose=1
    )
    
    # 5. Overwrite the active model
    model.save(model_path)
    print("Retraining complete. New model weights saved to production.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()
    
    if args.retrain:
        retrain_model()