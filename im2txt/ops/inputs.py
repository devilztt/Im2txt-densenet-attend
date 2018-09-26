# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Input ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np


def parse_sequence_example(serialized, image_feature, caption_feature):
  """Parses a tensorflow.SequenceExample into an image and caption.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.
    image_feature: Name of SequenceExample context feature containing image
      data.
    caption_feature: Name of SequenceExample feature list containing integer
      captions.

  Returns:
    encoded_image: A scalar string Tensor containing a JPEG encoded image.
    caption: A 1-D uint64 Tensor with dynamically specified length.
  """
  context, sequence = tf.parse_single_sequence_example(
      serialized,
      context_features={
          image_feature: tf.FixedLenFeature([], dtype=tf.string)
      },
      sequence_features={
          caption_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64),
      })

  encoded_image = context[image_feature]
  caption = sequence[caption_feature]
  return encoded_image, caption


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
  """Prefetches string values from disk into an input queue.

  In training the capacity of the queue is important because a larger queue
  means better mixing of training examples between shards. The minimum number of
  values kept in the queue is values_per_shard * input_queue_capacity_factor,
  where input_queue_memory factor should be chosen to trade-off better mixing
  with memory usage.

  Args:
    reader: Instance of tf.ReaderBase.
    file_pattern: Comma-separated list of file patterns (e.g.
        /tmp/train_data-?????-of-00100).
    is_training: Boolean; whether prefetching for training or eval.
    batch_size: Model batch size used to determine queue capacity.
    values_per_shard: Approximate number of values per shard.
    input_queue_capacity_factor: Minimum number of values to keep in the queue
      in multiples of values_per_shard. See comments above.
    num_reader_threads: Number of reader threads to fill the queue.
    shard_queue_name: Name for the shards filename queue.
    value_queue_name: Name for the values input queue.

  Returns:
    A Queue containing prefetched string values.
  """
  #1.产生文件名列表
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)#Prefetching values from 16 files matching /data/devilztt/image-caption/train-?????-of-00016

  if is_training:
    #2.生成一个先入先出的队列， 文件阅读器会需要它来读取数据。
    #string_input_producer 提供的可配置参数来设置文件名乱序和最大的训练迭代数
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=True, capacity=16, name=shard_queue_name)
    
    min_queue_examples = values_per_shard * input_queue_capacity_factor #16
    capacity = min_queue_examples + 100 * batch_size
    
    #创建一个随机队列，最大长度为capacity，出队后的最小长度为min_after_dequeue：
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)
  else:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=False, capacity=1, name=shard_queue_name)
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  enqueue_ops = []
  for _ in range(num_reader_threads):
    #通过传入的tfrecordreader 对filename_queue队列进行阅读解码，返回的是一个example 样本
    _, value = reader.read(filename_queue)
    #将得到的value传入随机队列，并添加到enqueue_ops
    enqueue_ops.append(values_queue.enqueue([value]))
  
  #QueueRunner会为每次迭代(epoch)将所有的文件名加入文件名队列中
  #tf.train.queue_runner.QueueRunner（）持有一个队列的入列操作列表
  #queue 队列 enqueue_ops 用于线程中运行的入列操作列表
  #add_queue_runner 增加一个queue_runner到graph的收集器
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops))
  tf.summary.scalar(
      "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue


def batch_with_dynamic_pad(images_and_captions,
                           batch_size,
                           queue_capacity,
                           add_summaries=False):
  """Batches input images and captions.

  This function splits the caption into an input sequence and a target sequence,
  where the target sequence is the input sequence right-shifted by 1. Input and
  target sequences are batched and padded up to the maximum length of sequences
  in the batch. A mask is created to distinguish real words from padding words.

  Example:
    Actual captions in the batch ('-' denotes padded character):
      [
        [ 1 2 3 4 5 ],
        [ 1 2 3 4 - ],
        [ 1 2 3 - - ],
      ]

    input_seqs:
      [
        [ 1 2 3 4 ],
        [ 1 2 3 - ],
        [ 1 2 - - ],
      ]

    target_seqs:
      [
        [ 2 3 4 5 ],
        [ 2 3 4 - ],
        [ 2 3 - - ],
      ]

    mask:
      [
        [ 1 1 1 1 ],
        [ 1 1 1 0 ],
        [ 1 1 0 0 ],
      ]

  Args:
    images_and_captions: A list of pairs [image, caption], where image is a
      Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
      any length. Each pair will be processed and added to the queue in a
      separate thread.
    batch_size: Batch size.
    queue_capacity: Queue capacity.
    add_summaries: If true, add caption length summaries.

  Returns:
    images: A Tensor of shape [batch_size, height, width, channels].
    input_seqs: An int32 Tensor of shape [batch_size, padded_length].
    target_seqs: An int32 Tensor of shape [batch_size, padded_length].
    mask: An int32 0/1 Tensor of shape [batch_size, padded_length].
  """
  enqueue_list = []
  for image, caption in images_and_captions:
    
    def true_fn():
      return tf.slice(caption, [0], [16])
    
    def false_fn():
      return caption
    
    caption_length = tf.shape(caption)[0]

    caption=tf.cond(caption_length>16,true_fn=true_fn,false_fn=false_fn)

    caption_length = tf.shape(caption)[0]


    input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)#先减1再扩充维度
    #对caption进行切片操作，
    input_seq = tf.slice(caption, [0], input_length)#tf.slice(inputs,begin,size,name='')
    target_seq = tf.slice(caption, [1], input_length)
    indicator = tf.ones(input_length, dtype=tf.int64)#tf.ones(shape, dtype=tf.float32, name=None)
    
    padding_lenth=tf.subtract(16,caption_length)
    padding=tf.zeros([padding_lenth],dtype=tf.int64)
    
    input_seq=tf.concat([input_seq,padding],0)
    target_seq=tf.concat([target_seq,padding],0)
    indicator=tf.concat([indicator,padding],0)
    # caption_length=len(caption)
    # input_length=caption_length-1
    # input_seq=caption[:input_length]
    # target_seq=caption[1:input_length]
    # indicator=np.ones([input_length])

    # input_seq=sequence.pad_sequences(input_seq, padding='post', maxlen=16)
    # target_seq=sequence.pad_sequences(target_seq, padding='post', maxlen=16)
    # indicator=sequence.pad_sequences(indicator, padding='post', maxlen=16)
    enqueue_list.append([image, input_seq, target_seq, indicator])

  images, input_seqs, target_seqs, mask = tf.train.batch_join(
      enqueue_list,
      batch_size=batch_size,
      capacity=queue_capacity,
      dynamic_pad=True,
      name="batch_and_pad")

  if add_summaries:
    lengths = tf.add(tf.reduce_sum(mask, 1), 1)
    tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths))
    tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths))
    tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths))

  return images, input_seqs, target_seqs, mask
