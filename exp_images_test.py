import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
import gc
# ignore all warnings and info
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import traceback
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub

from utils import crit_dividing, class_direction, img_model_links, all_heuristic_weight

gpus = tf.config.list_physical_devices('GPU')

if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

def experiment(
    train_ds,
    test_ds,
    model_link=None,
    epochs=5,
    num_train=64,
    num_classes=10,
    f=None):
  
  input_shape = (224, 224, 3) 

  try:
    base_model = hub.KerasLayer(
          model_link, 
          trainable=False)
  except:
    print(f'The pre-trained model not found from the kaggle server: {model_link}')
    print(f'The pre-trained model not found from the kaggle server: {model_link}', file=f)
    return
    
  # extract features to train
  model = Sequential([
    layers.Input(input_shape),
    layers.Rescaling(scale=1.0/255),
    base_model
  ])
  model.build(input_shape=(None, *input_shape))
  
  num_batches = num_train // 16
  X, labels = [], []
  for i, l in train_ds.take(num_batches):
    X.append(model(i).numpy())
    labels.append(l.numpy())

  X = np.concatenate(X, axis=0)  
  labels = np.concatenate(labels)

  if len(set(labels)) != num_classes:
    print(f'{num_train} training examples must have all class objects')
    print(f'{num_train} training examples must have all class objects', file=f)
    return

  # converting numpy allows quicker access indexing
  print("Output shape:", X.shape[1])
  print("Output shape:", X.shape[1], file=f)
  
  print('='*30, 'Heuristic weights calculation', '='*30)
  print('='*30, 'Heuristic weights calculation', '='*30, file=f)
  # calculate weights
  # get number of classes
  w = all_heuristic_weight(X, labels)
  return
  
  # remove the old references
  del model
  tf.keras.backend.clear_session()
  time.sleep(3)
  gc.collect()

  # build model again
  # extract features to train
  model = Sequential([
    layers.Input(input_shape),
    layers.Rescaling(scale=1.0/255),
    base_model,
    layers.Dense(num_classes, name='classifier')
  ])
  model.build(input_shape=(None, *input_shape))
  # copy initial weights to train again
  init_w = model.trainable_variables[-2]

  # assign heuristic weights
  model.trainable_variables[-2].assign(-w)
  model.compile(optimizer=keras.optimizers.legacy.Adam(
    learning_rate=1e-1, decay=1e-1),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
  print('='*30, 'Evaluation results on all heuristic weights', '='*30)
  print('='*30, 'Evaluation results on all heuristic weights', '='*30, file=f)
  res = model.evaluate(train_ds, verbose=False)
  print(f'Training loss: {res[0]:.4f}, accuracy: {res[1]:.4f}')
  print(f'Training loss: {res[0]:.4f}, accuracy: {res[1]:.4f}', file=f)
  res = model.evaluate(test_ds, verbose=False)
  print(f'Test loss: {res[0]:.4f}, accuracy: {res[1]:.4f}')
  print(f'Test loss: {res[0]:.4f}, accuracy: {res[1]:.4f}', file=f)
  
  
  if epochs > 0:
    tf.keras.backend.clear_session()
    time.sleep(3)
    gc.collect()
    print('='*30, f'Training model on all heurstic weights with epochs {epochs}', '='*30)
    print('='*30, f'Training model on all heurstic weights with epochs {epochs}', '='*30, file=f)
    callback = tf.keras.callbacks.EarlyStopping(
      monitor='loss',
      min_delta=1e-2,
      patience=1,
      restore_best_weights=True)
    hist = model.fit(
      train_ds, 
      epochs=epochs,
      callbacks=[callback])
    best_epoch = np.argmin(hist.history['loss'])
    res = (hist.history['loss'][best_epoch], hist.history['accuracy'][best_epoch])
    print(f'Training loss: {res[0]:.4f}, accuracy: {res[1]:.4f}')
    print(f'Training loss: {res[0]:.4f}, accuracy: {res[1]:.4f}', file=f)
    res = model.evaluate(test_ds, verbose=False)
    print(f'Test loss: {res[0]:.4f}, accuracy: {res[1]:.4f}')
    print(f'Test loss: {res[0]:.4f}, accuracy: {res[1]:.4f}', file=f)

    tf.keras.backend.clear_session()
    time.sleep(3)
    gc.collect()
    model.compile(optimizer=keras.optimizers.legacy.Adam(
      learning_rate=1e-2, decay=1e-2),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
    print('='*30, f'Training model on all initial weights with epochs {epochs}', '='*30)
    print('='*30, f'Training model on all initial weights with epochs {epochs}', '='*30, file=f)
    # reassign old weights
    model.trainable_variables[-2].assign(init_w)
    hist = model.fit(
      train_ds, 
      epochs=epochs,
      callbacks=[callback])
    best_epoch = np.argmin(hist.history['loss'])
    res = (hist.history['loss'][best_epoch], hist.history['accuracy'][best_epoch])
    print(f'Training loss: {res[0]:.4f}, accuracy: {res[1]:.4f}')
    print(f'Training loss: {res[0]:.4f}, accuracy: {res[1]:.4f}', file=f)
    res = model.evaluate(test_ds, verbose=False)
    print(f'Test loss: {res[0]:.4f}, accuracy: {res[1]:.4f}')
    print(f'Test loss: {res[0]:.4f}, accuracy: {res[1]:.4f}', file=f)

  # remove the old references
  del model
  tf.keras.backend.clear_session()
  time.sleep(3)
  gc.collect()

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--tr-folder', type=str)
  parser.add_argument('--te-folder', type=str, default=None)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--epochs', type=int, default=5)
  parser.add_argument('--model', type=str, choices=list(img_model_links.keys()))
  parser.add_argument('--output', type=str, default='output.txt')

  args = parser.parse_args()

  tr_folder = args.tr_folder
  te_folder = args.te_folder
  seed = args.seed
  model_name = args.model

  if seed:
    tf.random.set_seed(seed=seed)

  with open(model_name + '-' + args.output, 'a+') as f:

    if te_folder:
      # load datasets
      print("Training dataset: ")
      print("Training dataset: ", file=f)
      train_ds = tf.keras.utils.image_dataset_from_directory(
          tr_folder,
          image_size=(224, 224),
          batch_size=16)
      print("Test dataset: ")
      print("Test dataset: ", file=f)
      test_ds = tf.keras.utils.image_dataset_from_directory(
          te_folder,
          image_size=(224, 224),
          batch_size=16)
    else:
      print("Training dataset: ")
      print("Training dataset: ", file=f)
      train_ds = tf.keras.utils.image_dataset_from_directory(
          tr_folder,
          image_size=(224, 224),
          batch_size=16,
          validation_split=0.2,
          seed=seed,
          subset='training')
      print("Test dataset: ")
      print("Test dataset: ", file=f)
      test_ds = tf.keras.utils.image_dataset_from_directory(
          tr_folder,
          image_size=(224, 224),
          batch_size=16,
          validation_split=0.2,
          seed=seed,
          subset='validation')
    num_classes = len(train_ds.class_names)
    # to test code
    experiment(
      train_ds=train_ds,
      test_ds=test_ds,
      model_link="https://kaggle.com/models/google/mobilenet-v3/frameworks/TensorFlow2/variations/large-075-224-classification/versions/1",
      num_train=128,
      num_classes=num_classes,
      epochs=args.epochs,
      f=f)
    return

    f.write(f'Now: {datetime.datetime.now()}\n')
    f.write(f'Dataset: {tr_folder}\n{te_folder}\n')
    print("="*30, "Running all image classification problems", "="*30)
    print("="*30, "Running all image classification problems", "="*30, file=f)
    print('\n', "*"*30, f"Model name: {model_name}", "*"*30)
    print('\n', "*"*30, f"Model name: {model_name}", "*"*30, file=f)
    for version, link in img_model_links[model_name].items():
      print("\nModel version:", version)
      print("\nModel version:", version, file=f)
      # Please do not change these numbers
      # for num_train in [128]: # for test
      for num_train in [64, 128, 256, 512, 1024, 2048]:
        print("\n", f"Number of examples in computing weights heuristically: {num_train}")
        print("\n", f"Number of examples in computing weights heuristically: {num_train}", file=f)
        try:
          experiment(
            train_ds=train_ds,
            test_ds=test_ds,
            model_link=link,
            num_train=num_train,
            num_classes=num_classes,
            epochs=args.epochs,
            f=f)
        except:
          print("A error occured")
          print("A error occured", file=f)
          traceback.print_exc()

if __name__ == '__main__':
  main()