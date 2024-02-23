import numpy as np
import tensorflow as tf

def entropy(counts):
  """
  counts - n vector
  """
  probs = counts / np.sum(counts)
  log = np.log(probs)
  return -np.sum(probs * log)

def gini(counts):
  """
  counts - n vector
  """
  probs = counts / np.sum(counts)
  return 1 - np.sum(probs ** 2) 

def weight_nikolay(u, **kwargs):
  """
  u - 2x2 matrix consisting of statistics: 
  the first index for intervals
  the second index for classess
  **kwargs - stores specific parameters for this weighting criteria
  m1-scalar
  m2-scalar
  num_labels - number of samples in each class
  """
  m1 = kwargs['m1']
  m2 = kwargs['m2']
  num_labels = kwargs['num_labels']
  sum1 = 0.0
  sum2 = 0.0
  # Furmulation
  for l in range(0, 2):
    for z in range(0, 2):
      sum1 += u[l, z] * (u[l, z] - 1)
      sum2 += u[l, z] * (num_labels[1 - z] - u[l, 1 - z])
  
  return (sum1 / m1) * (sum2 / m2)

def weight_entropy(u, **kwargs):
  """
  u - 2x2 matrix consisting of statistics: 
  the first index for intervals
  the second index for classess
  **kwargs - stores specific parameters for this weighting criteria
  tar_entropy - a scalar of target entropy
  n_objs - a scalar of number of samples
  """
  tar_entropy = kwargs['tar_entropy']
  n_objs = kwargs['n_objs']
  f_entropy = entropy(u[0]) * (u[0, 0] + u[0, 1]) + \
    entropy(u[1]) * (u[1, 0] + u[1, 1])
  f_entropy /= n_objs
  weight = tar_entropy - f_entropy
  return weight

def weight_gini(u, **kwargs):
  """
  u - 2x2 matrix consisting of statistics: 
  the first index for intervals
  the second index for classess
  **kwargs - stores specific parameters for this weighting criteria
  tar_gini - a scalar of target gini
  n_objs - a scalar of number of samples
  """
  # the first interval
  # very pure implementation
  tar_gini = kwargs['tar_gini']
  n_objs = kwargs['n_objs']
  f_gini = gini(u[0]) * (u[0, 0] + u[0, 1]) + \
      gini(u[1]) * (u[1, 0] + u[1, 1])
  f_gini /= n_objs
  weight = tar_gini - f_gini

  return weight

def crit_dividing(x, y, 
                  sort_indices, 
                  num_labels,
                  wtype='nikolay'):
    
    # kwargs for information gain criteria
    weight_crit = weight_nikolay
    kwargs = {}
    wtype = wtype.lower()
    if wtype == 'nikolay':
      # dividing coefs
      kwargs['num_labels'] = num_labels
      kwargs['m1'] = np.sum(num_labels * (num_labels - 1))
      kwargs['m2'] = 2 * num_labels[0] * num_labels[1]
    elif wtype == 'entropy':
      kwargs['n_objs'] = y.shape[0]
      kwargs['tar_entropy'] = entropy(num_labels)
      weight_crit = weight_entropy
    elif wtype == 'gini':
      kwargs['n_objs'] = y.shape[0]
      kwargs['tar_gini'] = gini(num_labels)
      weight_crit = weight_gini
    else:
       raise NotImplementedError()

    # u[x, y] is count of x index of interval and y class. x is index of interval's and y is index of class. Where are x = {0, 1}.
    u = np.zeros((2, 2))
    # Assume all objects in the second interval, but it is not true in general
    u[1, 0] = num_labels[0]
    u[1, 1] = num_labels[1]
    # Result values
    # Default of max value is 0
    max_value = 0
    opt_index = sort_indices[0]

    min_index = sort_indices[0]
    max_index = sort_indices[-1]
    
    # for begin from 0 to len(vaules) - 1, beacuse each interval need min one object
    i = 0
    while i < len(x) - 1:
        # Count object's in fisrt interval by class
        u[0, y[sort_indices[i]]] += 1

        # Decrease the number of objects in the second interval
        u[1, y[sort_indices[i]]] -= 1

        # if current object and next object aren't equal
        if x[sort_indices[i]] != x[sort_indices[i + 1]]:
            current_val = weight_crit(u, **kwargs)
            # Check current max than more max value
            if current_val > max_value:
                max_value = current_val
                opt_index = sort_indices[i]
        i += 1
    return max_value, min_index, opt_index, max_index

def class_direction(x, y, border, num_obj_train):

    # Find feature direction
    cond = x <= border
    _, num_obj_feat = np.unique(y[cond], return_counts=True)

    pers =  num_obj_feat / num_obj_train
    norm_pers = pers / sum(pers)

    t = -1
    # The first interval is survived patient interval
    if norm_pers[0] > 0.5:
        # prediction validation set
        t = 1

    return t

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  multiples = [1 for _ in spectrogram.shape]
  multiples.append(3)
  spectrogram = spectrogram[..., tf.newaxis]
  spectrogram = tf.tile(spectrogram, multiples=multiples)
  return spectrogram

def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

# vectorized versions: not implemented yet
# to be implemented
def heuristic_weight(x, y, m1, m2, n_ls):
  n_obs = x.shape[0]
  n_cls = y.shape[1]
  inds  = np.argsort(x)

  # counts of objects in two intervals and one vs all over classes
  u = np.zeros((2, 2, n_cls), dtype=int)
  # initially, all objects in the second interval
  u[1, :, :] = n_ls.T.copy()
  # for computing class directions
  u_opt = u.copy()
  # default optimal values
  opt_vals = np.zeros(n_cls)

  for i in range(n_obs - 1):
    # since we including objects into the first interval
    # we increase the first interval count values
    print(u[0])
    print(inds[i], y[inds[i]])

    u[0, y[inds[i]]] += 1
    
    print(u[0])
    break

    # while decrease the second interval count values
    u[1, y[inds[i]]] -= 1
    # two objects in a row must be equal
    if x[inds[i]] != x[inds[i + 1]]:
      # main equation
      p1 = np.sum(u * (u - 1))
      p2 = 2 * (u[0, 0] * u[1, 1] + u[1, 0] * u[0, 1])
      vals = (p1 / m1) * (p2 / m2)

      # updating optimal values
      conds = vals > opt_vals
      opt_vals[conds] = vals[conds]
      u_opt[:, :, conds] = u[:, :, conds]
  
  u_opt = 1. / n_cls * u_opt
  prob = u_opt / np.sum(u_opt, axis=0)
  conds = prob[0, 0, :] > 0.5
  opt_vals[conds] *= -1

  return opt_vals

def all_heuristic_weight(X, y):
   
  n_obs, n_fs = X.shape
  n_cls = np.max(y) + 1
  # be very carefull, assuming 1 is target class
  # since one-hot encoding
  one_hot_y = np.zeros((n_obs, n_cls), dtype=int)
  one_hot_y[np.arange(n_obs), y] = 1
  n_ls = np.zeros((n_cls, 2), dtype=int)
  n_ls[:, 1] = np.sum(one_hot_y, axis=0)
  n_ls[:, 0] = n_obs - n_ls[:, 1]

  m1 = np.sum(n_ls * (n_ls - 1), axis=1)
  m2 = 2 * n_ls[:, 0] * n_ls[:, 1]

  w = np.zeros((n_fs, n_cls))
  for i in range(n_fs):
    x = X[:, i]
    w[i] = heuristic_weight(x, one_hot_y, m1, m2, n_ls)
    break
  
  return w

img_model_links = {
   'mobilenet_v3': {
      'large-075-224-classification': "https://kaggle.com/models/google/mobilenet-v3/frameworks/TensorFlow2/variations/large-075-224-classification/versions/1",
      'small-075-224-classification': "https://www.kaggle.com/models/google/mobilenet-v3/frameworks/TensorFlow2/variations/small-075-224-classification/versions/1",
      'small-075-224-feature-vector': "https://www.kaggle.com/models/google/mobilenet-v3/frameworks/TensorFlow2/variations/small-075-224-feature-vector/versions/1",
    },

    'efficientnet': {
       'b0-classification': 'https://www.kaggle.com/models/tensorflow/efficientnet/frameworks/TensorFlow2/variations/b0-classification/versions/1',
       'b0-feature-vector': 'https://www.kaggle.com/models/tensorflow/efficientnet/frameworks/TensorFlow2/variations/b0-feature-vector/versions/1',
       'b1-classification': 'https://www.kaggle.com/models/tensorflow/efficientnet/frameworks/TensorFlow2/variations/b1-classification/versions/1',
       'b1-feature-vector': 'https://www.kaggle.com/models/tensorflow/efficientnet/frameworks/TensorFlow2/variations/b1-feature-vector/versions/1',
       'b4-classification': 'https://www.kaggle.com/models/tensorflow/efficientnet/frameworks/TensorFlow2/variations/b4-classification/versions/1',
       'b4-feature-vector': 'https://www.kaggle.com/models/tensorflow/efficientnet/frameworks/TensorFlow2/variations/b4-feature-vector/versions/1'
    },
    'inception_v3': {
        'classification': "https://www.kaggle.com/models/google/inception-v3/frameworks/TensorFlow2/variations/classification/versions/2",
        'feature-vector': "https://www.kaggle.com/models/google/inception-v3/frameworks/TensorFlow2/variations/feature-vector/versions/2",
    },
    'resnet_50': {
        'classification': "https://www.kaggle.com/models/tensorflow/resnet-50/frameworks/TensorFlow2/variations/classification/versions/1",
        'feature-vector': "https://www.kaggle.com/models/tensorflow/resnet-50/frameworks/TensorFlow2/variations/feature-vector/versions/1",
    },
    'vision_transformer': {
        'vit-b16-classification': "https://www.kaggle.com/models/spsayakpaul/vision-transformer/frameworks/TensorFlow2/variations/vit-b16-classification/versions/1",
        'vit-b16-fe': 'https://www.kaggle.com/models/spsayakpaul/vision-transformer/frameworks/TensorFlow2/variations/vit-b16-fe/versions/1',
    },
    'convnext': {
        'base-1k-224': 'https://www.kaggle.com/models/spsayakpaul/convnext/frameworks/TensorFlow2/variations/base-1k-224/versions/1',
    },
    'mlp-mixer': {
        'mixer-b16-i1k-classification': "https://www.kaggle.com/models/spsayakpaul/mlp-mixer/frameworks/TensorFlow2/variations/mixer-b16-i1k-classification/versions/1",
        'mixer-b16-i1k-fe': "https://www.kaggle.com/models/spsayakpaul/mlp-mixer/frameworks/TensorFlow2/variations/mixer-b16-i1k-fe/versions/1",
        'mixer-b32-sam-classification': "https://www.kaggle.com/models/spsayakpaul/mlp-mixer/frameworks/TensorFlow2/variations/mixer-b32-sam-classification/versions/1",
        'mixer-b32-sam-fe': "https://www.kaggle.com/models/spsayakpaul/mlp-mixer/frameworks/TensorFlow2/variations/mixer-b32-sam-fe/versions/1"
    }
}

text_model_links = {
    'small_bert': {
      'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
      'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
      'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
      'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    },
    'bert': {
      'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
      'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
      'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    },
    'albert': {
      'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    },
    'electra':{
      'electra_small':
        'https://tfhub.dev/google/electra_small/2',
      'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    },
    'experts':{
      'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
      'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    },
    'talking-heads': {
      'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
    }
}

text_prep_links = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
}