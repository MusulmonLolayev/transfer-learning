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

from utils import crit_dividing, class_direction, img_model_links
from utils import squeeze, make_spec_ds


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
    f=None,
    wtype='nikolay',
    only_sign=False,
    wthreshold=0.0):
  
  input_shape = (124, 129, 3) 

  try:
    base_model = hub.KerasLayer(
          model_link, 
          trainable=False)
  except:
    print(f'The pre-trained model not found from the kaggle server: {model_link}')
    print(f'The pre-trained model not found from the kaggle server: {model_link}', file=f)
    return
  
  # Instantiate the `tf.keras.layers.Normalization` layer.
  norm_layer = layers.Normalization()
  # Fit the state of the layer to the spectrograms
  # with `Normalization.adapt`.
  norm_layer.adapt(data=train_ds.map(map_func=lambda spec, label: spec))

  # extract features to train
  model = Sequential([
    layers.Input(input_shape),
    layers.Resizing(224, 224),
    norm_layer,
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
  w = np.zeros((X.shape[1], num_classes))

  for i in range(X.shape[1]):
    x = X[:, i]
    sort_indices = np.argsort(x)
    for j in range(num_classes):
      y = (labels != j).astype(int)

      _, num_labels = np.unique(y, return_counts=True)
      res = crit_dividing(x, y, sort_indices, num_labels, wtype)
      sign = class_direction(x, y, x[int(res[2])], num_labels)
      if only_sign:
        w[i, j] = 0.2 * sign
      else:
        w[i, j] = 0 if wthreshold > res[0] else res[0] * sign
  
  # remove the old references
  del model
  tf.keras.backend.clear_session()
  time.sleep(3)
  gc.collect()

  # build model again
  # extract features to train
  # extract features to train
  model = Sequential([
    layers.Input(input_shape),
    layers.Resizing(224, 224),
    norm_layer,
    base_model,
    layers.Dense(num_classes, name='classifier')
  ])
  model.build(input_shape=(None, *input_shape))
  # copy initial weights to train again
  init_w = model.trainable_variables[-2]

  # assign heuristic weights
  model.trainable_variables[-2].assign(-w)
  model.compile(optimizer=keras.optimizers.legacy.Adam(),
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
    model.compile(optimizer=keras.optimizers.legacy.Adam(),
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
  parser.add_argument('--only-sign', type=bool, default=False)
  parser.add_argument('--w-threshold', type=float, default=0)
  parser.add_argument('--w-type', type=str, choices=['nikolay', 'entropy', 'gini'])
  parser.add_argument('--output', type=str, default='output.txt')

  args = parser.parse_args()

  tr_folder = args.tr_folder
  te_folder = args.te_folder
  only_sign = args.only_sign
  wthreshold = args.w_threshold
  seed = args.seed
  model_name = args.model
  wtype = args.w_type
  out_name: str = args.output
  if not out_name.endswith('.txt'):
    out_name += '.txt'

  if seed:
    tf.random.set_seed(seed=seed)
  log_f_path = f'res-logs/{model_name}-{wtype}-{only_sign}-{wthreshold:.2f}-{out_name}'
  with open(log_f_path, 'w') as f:

    if te_folder:
      # load datasets
      print("Training dataset: ")
      print("Training dataset: ", file=f)
      train_ds = tf.keras.utils.audio_dataset_from_directory(
          tr_folder,
          batch_size=16,
          output_sequence_length=16000)
      print("Test dataset: ")
      print("Test dataset: ", file=f)
      test_ds = tf.keras.utils.audio_dataset_from_directory(
          te_folder,
          batch_size=16,
          output_sequence_length=16000)
    else:
      print("Training dataset: ")
      print("Training dataset: ", file=f)
      train_ds = tf.keras.utils.audio_dataset_from_directory(
          tr_folder,
          batch_size=16,
          validation_split=0.2,
          seed=seed,
          subset='training',
          output_sequence_length=16000)
      print("Test dataset: ")
      print("Test dataset: ", file=f)
      test_ds = tf.keras.utils.audio_dataset_from_directory(
          tr_folder,
          batch_size=16,
          validation_split=0.2,
          seed=seed,
          subset='validation',
          output_sequence_length=16000)
    num_classes = len(train_ds.class_names)
    
    # preprocessing audio dataset
    # remove the extra axis
    # (TensorSpec(shape=(None, 16000, None), dtype=tf.float32, name=None),
    # TensorSpec(shape=(None,), dtype=tf.int32, name=None))
    

    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    test_ds = test_ds.map(squeeze, tf.data.AUTOTUNE)

    # convert to spectrograms
    train_spec_ds = make_spec_ds(train_ds)
    test_spec_ds = make_spec_ds(test_ds)

    # reduce read latency
    train_spec_ds = train_spec_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    test_spec_ds = test_spec_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)

    # # to test code
    # experiment(
    #   train_ds=train_spec_ds,
    #   test_ds=test_spec_ds,
    #   model_link="https://kaggle.com/models/google/mobilenet-v3/frameworks/TensorFlow2/variations/large-075-224-classification/versions/1",
    #   num_train=128,
    #   num_classes=num_classes,
    #   epochs=args.epochs,
    #   f=f,
    #   wthreshold=wthreshold,
    #   wtype=wtype,
    #   only_sign=only_sign)
    # return

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
            train_ds=train_spec_ds,
            test_ds=test_spec_ds,
            model_link=link,
            num_train=num_train,
            num_classes=num_classes,
            epochs=args.epochs,
            f=f,
            wthreshold=wthreshold,
            wtype=wtype,
            only_sign=only_sign)
        except:
          print("A error occured")
          print("A error occured", file=f)
          traceback.print_exc()

if __name__ == '__main__':
  main()