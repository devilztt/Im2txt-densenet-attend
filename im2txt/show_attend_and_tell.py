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

"""Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

"Show and Tell: A Neural Image Caption Generator"
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from im2txt.ops import image_embedding
from im2txt.ops import image_processing
from im2txt.ops import inputs as input_ops
import pickle

class ShowAndTellModel(object):
  """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

  def __init__(self, config, mode, train_inception=False):
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_inception: Whether the inception submodel variables are trainable.
    """
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode
    self.train_inception = train_inception

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    self.weight_initializer=tf.contrib.layers.xavier_initializer()

    self.const_initializer = tf.constant_initializer(0.0)

    self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.input_seqs = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.target_seqs = None

    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.input_mask = None

    # A float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings = None

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    self.seq_embeddings = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_losses = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_loss_weights = None

    # Collection of variables from the inception submodel.
    self.densenet_variable = []

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None
    
    #vocab的长度 embedding的第一维
    self.V = None
    
    #特征图的第二维
    self.L = 196
    
    #特征图的第三维
    self.D = 384
    
    #embedding的第二维
    self.M = 384
    
    #隐层的大小
    self.H = 512

    self.alpha_c=1.0
    
    #时间步长
    self.T = config.number_step

    self.selector=True

    self.ctx2out=True

    self.prev2out=True

    self.embedding_map=pickle.load(open(config.embedding_file,'rb'))
  
  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def process_image(self, encoded_image, thread_id=0):
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions.

    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return image_processing.process_image(encoded_image,
                                          is_training=self.is_training(),
                                          height=self.config.image_height,
                                          width=self.config.image_width,
                                          thread_id=thread_id,
                                          image_format=self.config.image_format)

  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")

      # Process image and insert batch dimensions.
      images = tf.expand_dims(self.process_image(image_feed), 0)
      input_seqs = tf.expand_dims(input_feed, 1)

      # No target sequences or input mask in inference mode.
      target_seqs = None
      input_mask = None
    else:
      # Prefetch serialized SequenceExample protos.
      # 存有example的队列
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          is_training=self.is_training(),
          batch_size=self.config.batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert self.config.num_preprocess_threads % 2 == 0
      images_and_captions = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        #获取图片和描述
        encoded_image, caption = input_ops.parse_sequence_example(
            serialized_sequence_example,
            image_feature=self.config.image_feature_name,
            caption_feature=self.config.caption_feature_name)
        #对图片进行预处理
        image = self.process_image(encoded_image, thread_id=thread_id)
        images_and_captions.append([image, caption])

      # Batch inputs.
      queue_capacity = (2 * self.config.num_preprocess_threads *
                        self.config.batch_size)
      #组合成一个batch
      images, input_seqs, target_seqs, input_mask = (
          input_ops.batch_with_dynamic_pad(images_and_captions,
                                           batch_size=self.config.batch_size,
                                           queue_capacity=queue_capacity,
                                           num_step=self.config.number_step))

    self.images = images
    self.input_seqs = input_seqs
    self.target_seqs = target_seqs
    self.input_mask = input_mask

  def build_image_embeddings(self):
    """Builds the image model subgraph and generates image embeddings.

    Inputs:
      self.images

    Outputs:
      self.image_embeddings
    """

    #用densenet生成一个imageembedding
    densenet_out = image_embedding.densenet_161(
        self.images,
        trainable=self.train_inception,
        is_training=self.is_training())
    self.densenet_variable = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="densenet161")



    # Save the embedding size in the graph.
    self.embedding_size=384
    tf.constant(self.embedding_size, name="embedding_size")

    self.features = densenet_out
    self.L = 196
    self.D = 384
  
  def build_seq_embeddings(self):
    """Builds the input sequence embeddings.

    Inputs:
      self.input_seqs

    Outputs:
      self.seq_embeddings
    """
    # with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
    #   #这里好像是随机初始化的embedding_map？ 
    #   embedding_map = tf.get_variable(
    #       name="map",
    #       shape=[self.config.vocab_size, self.embedding_size],
    #       initializer=self.initializer)
      #返回的是seq的向量列表，也就是说input seq是一个index列表
    embedding_map=self.embedding_map
    seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

    self.seq_embeddings = seq_embeddings
    self.V=self.config.vocab_size
    # self.M=self.embedding_size
  
  def _batch_norm(self, x, mode='train', name=None):
      return tf.contrib.layers.batch_norm(inputs=x,
                                          decay=0.95,
                                          center=True,
                                          scale=True,
                                          is_training=(mode=='train'),
                                          updates_collections=None,
                                          scope=(name+'batch_norm'))

  def _get_initial_lstm(self, features):
      with tf.variable_scope('initial_lstm'):
          features_mean = tf.reduce_mean(features, 1)

          w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
          b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
          h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

          w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
          b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
          c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
          return c, h

  def _project_features(self, features):
      with tf.variable_scope('project_features'):
          w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
          features_flat = tf.reshape(features, [-1, self.D])
          features_proj = tf.matmul(features_flat, w)
          features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
          return features_proj

  #生成attention的向量，结合上一个t和特征图的features得到
  def _attention_layer(self, features, features_proj, h, reuse=False):
      with tf.variable_scope('attention_layer', reuse=reuse):
          w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
          b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
          w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

          h_att = tf.nn.elu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
          out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
          alpha = tf.nn.softmax(out_att)
          context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
          return context, alpha
  
  def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
      with tf.variable_scope('logits', reuse=reuse):
          w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
          b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
          # w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
          # b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

          if dropout:
              h = tf.nn.dropout(h, 0.7)
          h_logits = tf.matmul(h, w_h) + b_h # self.M

          if self.ctx2out:
              w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
              h_logits += tf.matmul(context, w_ctx2out)#self.M

          if self.prev2out:
              h_logits += x
          h_logits = tf.nn.tanh(h_logits)

          if dropout:
              h_logits = tf.nn.dropout(h_logits, 0.7)
          # out_logits = tf.matmul(h_logits, w_out) + b_out
          return h_logits

  def _selector(self, context, h, reuse=False):
      with tf.variable_scope('selector', reuse=reuse):
          w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
          b = tf.get_variable('b', [1], initializer=self.const_initializer)
          beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
          context = tf.multiply(beta, context, name='selected_context')
          return context, beta
  

  def build_model(self):
    """Builds the model.

    Inputs:
      self.image_embeddings
      self.seq_embeddings
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)

    Outputs:
      self.total_loss (training and eval only)
      self.target_cross_entropy_losses (training and eval only)
      self.target_cross_entropy_loss_weights (training and eval only)
    """
    # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
    # modified LSTM in the "Show and Tell" paper has no biases and outputs
    # new_c * sigmoid(o).
    features=self.features
    captions_out = self.target_seqs
    mask = self.input_mask
    x=self.seq_embeddings #经过词嵌入后seq序列
    with tf.variable_scope('lstm'):
     
      features = self._batch_norm(features, mode=self.mode, name='conv_features') #将特征向量归一化
                
      features_proj = self._project_features(features=features) #经过一个全连接 生成e0

      alpha_list = []

      lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H) #定义一个lstm单元
      
      if self.mode == "train":
        lstm_cell = tf.contrib.rnn.DropoutWrapper(
            lstm_cell,
            input_keep_prob=self.config.lstm_dropout_keep_prob,
            output_keep_prob=self.config.lstm_dropout_keep_prob)

      
      c, h = self._get_initial_lstm(features=features) #将特征图在196这个维度求和平均得到一个1*512的特征向量初始话c 和 h

      if self.mode == "inference":
        # In inference mode, use concatenated states for convenient feeding and
        # fetching.
        tf.concat(axis=1, values=(c,h), name="initial_state") #得到初始的state

        # Placeholder for feeding a batch of concatenated states.
        state_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(lstm_cell.state_size)],
                                    name="state_feed") #输入每个句子的state
        state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

        
        context, alpha = self._attention_layer(features, features_proj, state_tuple[1])#将features和h传入attention层，h代表当前状态，关注的是原图哪个区域
        

        if self.selector:
            context, beta = self._selector(context, state_tuple[1])

        with tf.variable_scope('lstm_cell'):
            _, (c, h) = lstm_cell(inputs=tf.concat([tf.squeeze(x,1), context],1),state=state_tuple)

        lstm_outputs = self._decode_lstm(x, h, context, dropout=False)
        # Concatentate the resulting state.
        tf.concat(axis=1, values=(c,h), name="state")
      else:
        # loss=0.0
        lstm_outputs=[]
        #lstm_logits=[]
        #print(x.get_shape()[1])
        #sequence_length = x.get_shape()[1]
        # sequence_length = tf.reduce_sum(self.input_mask, 1)
        # print(sequence_length.get_shape())

        for t in range(self.T-1):
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))#将features和h传入attention层，h代表当前状态，关注的是原图哪个区域
            alpha_list.append(alpha)#将得到的alpha添加到列表

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))

            with tf.variable_scope('lstm_cell', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat( [x[:,t,:], context],1), state=[c, h])

            logits = self._decode_lstm(x[:,t,:], h, context, dropout=True, reuse=(t!=0))

            #lstm_logits.append(logits)
            #loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=captions_out[:, t],logits=logits)*tf.to_float(mask[:, t]))
            lstm_outputs.append(logits)

        #self.total_loss=loss / tf.to_float(self.config.batch_size)

      #Stack batches vertically.
        #print(lstm_outputs.get_shape())
      lstm_outputs = tf.reshape(lstm_outputs, [-1, self.M])
    # # lstm_logits=tf.reshape(lstm_logits,[-1,tf.shape(logits)])
    
      with tf.variable_scope("logits") as logits_scope:
        logits = tf.contrib.layers.fully_connected(
            inputs=lstm_outputs,
            num_outputs=self.config.vocab_size,
            activation_fn=None,
            weights_initializer=self.initializer,
            scope=logits_scope)

      if self.mode == "inference":
        tf.nn.softmax(logits, name="softmax")
      else:
        if self.alpha_c > 0:
            alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
            alphas_all = tf.reduce_sum(alphas, 1)      # (N, L)
            alpha_reg = self.alpha_c * tf.reduce_sum((16./196 - alphas_all) ** 2)


        targets = tf.reshape(captions_out, [-1])
        weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

        # Compute losses.
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,logits=logits)+alpha_reg
        batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                            tf.reduce_sum(weights),
                            name="batch_loss")
        tf.losses.add_loss(batch_loss)
        total_loss = tf.losses.get_total_loss()

        # Add summaries.
        tf.summary.scalar("losses/batch_loss", batch_loss)
        tf.summary.scalar("losses/total_loss", total_loss)
        for var in tf.trainable_variables():
          tf.summary.histogram("parameters/" + var.op.name, var)

        self.total_loss = total_loss
        self.target_cross_entropy_losses = losses  # Used in evaluation.
        self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

  def setup_inception_initializer(self):
    """Sets up the function to restore inception variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
      saver = tf.train.Saver(self.densenet_variable)

      def restore_fn(sess):
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
                        self.config.inception_checkpoint_file)
        saver.restore(sess, self.config.inception_checkpoint_file)

      self.init_fn = restore_fn

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_image_embeddings()
    self.build_seq_embeddings()
    self.build_model()
    self.setup_inception_initializer()
    self.setup_global_step()
