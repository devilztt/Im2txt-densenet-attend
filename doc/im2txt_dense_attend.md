# Im2txt with densenet and attention

> 本项目在google Tensorflow中 的im2txt和show attend and tell 论文的基础上，将inception网络替换为densenet，作为CNN编码器，然后加入了自己理解的attention机制，



[TOC]

## 数据集准备

**数据集下载**

由于本人电脑资源有限，这里使用Flickr8K数据集作为项目数据集，下载地址如下  

[http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)

**数据集分析**  

下载的数据集分为两部分，一个是图片文件夹，另外一个是文档文件夹，文档文件夹中有关于图片的描述内容，一个图片有5句描述，还有一个打分情况，但是这个我没有用上，并且已经分好了train val 和test。其中train图片6000张，val图片1000张，test图片1000张（其实感觉可以把test的数据集取消掉，增加到Train里面去）总计8000张图片，一共有40460条描述（不知道为啥好像和图片数量没对齐）

**tfrecord制作**

> 采用tensorflow标准的方式制作tfrecord形式的数据集，便于训练时快速读取

数据集的制作参考的google im2txt里面的 build_mscoco_data.py  

根据Flickr8k.token.txt生成id对多个描述的字典，其中id对应的图片文件名

```python
def dict_id_to_cap(captions_file):
  id_to_captions={}
  with open(captions_file,'r') as f2 :
      for line in f2:
          st=line.strip().split('.',1)
          img_id=st[0]
          cap=st[1].split('\t')[1]
          id_to_captions.setdefault(img_id,[])
          id_to_captions[img_id].append(cap)
  return id_to_captions

```

根据Flickr_8k.trainImages.txt（存储的是训练用的图片文件名）从图片文件夹中读取训练用的图片数据，val和test操作一样


```python
  image_metadata = []
  num_captions = 0
  for image_id, base_filename in id_to_filename:
    filename = os.path.join(image_dir, base_filename)
    captions = [_process_caption(c) for c in id_to_captions[image_id]]
    image_metadata.append(ImageMetadata(image_id, filename, captions))
    num_captions += len(captions)
```

返回的是image_metadata的集合，里面存储的是ImageMetadata实例对象，一个ImageMetadata包含了图片id，图片路径，以及该图片对应的语句描述，并对captions进行处理，加上语句开始和结束符号

```python
def _process_caption(caption):
  tokenized_caption = [FLAGS.start_word]
  tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
  tokenized_caption.append(FLAGS.end_word)
  return tokenized_caption
```

根据所有的captions创建vocabulary，将caption中出现的词按照出现频率进行排序，并加上index，将生成的字典保存到本地文件夹中

```python
  counter = Counter()
  for c in captions:
    counter.update(c)
  print("Total words:", len(counter))

  # Filter uncommon words and sort by descending count.
  word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
  word_counts.sort(key=lambda x: x[1], reverse=True)
  print("Words in vocabulary:", len(word_counts))

  # Write out the word counts file.
  with tf.gfile.FastGFile(FLAGS.word_counts_output_file, "w") as f:
    f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
  print("Wrote vocabulary file:", FLAGS.word_counts_output_file)

  # Create the vocabulary dictionary.
  reverse_vocab = [x[0] for x in word_counts]
  unk_id = len(reverse_vocab)
  vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
  vocab = Vocabulary(vocab_dict, unk_id)
```

根据前面得到的image_metadata集合，将其中每一个ImageMetadata对象实例中每张图片对应的5个描述分离，让一张图对应一句描述，并形成新的images，images仍然是一个ImageMetadata的集合，然后对这个list进行随机打乱

```python
  images = [ImageMetadata(image.image_id, image.filename, [caption])
            for image in images for caption in image.captions]
  random.seed(12345)
  random.shuffle(images)
```

谷歌原文中用了多线程的方式对数据集进行切分，本文按照谷歌的方式，将训练集分为16份，每一份用一个线程去处理图片（将图片读取并进行jpg decode）存在内存中

生成tfrecord写入流

```python
    output_file = os.path.join(FLAGS.output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)
```

将每一个image根据其中的图片路径名读取出图片，再根据decode进行解码，将解码后的image存入features，由于caption不等长，所以存在featurelist中，这里根据前面生成的vocab，将每个caption里面的词转为词对应index，最后将两者写入SequenceExample中

```python
with tf.gfile.FastGFile(image.filename, "rb") as f:
    encoded_image = f.read()
  try:
    decoder.decode_jpeg(encoded_image)#解码
  except (tf.errors.InvalidArgumentError, AssertionError):
    print("Skipping file with invalid JPEG data: %s" % image.filename)
    return

  context = tf.train.Features(feature={
      "image/image_id": _str2bytes_feature(image.image_id),
      "image/data": _bytes_feature(encoded_image),
  })#将image_id 和 解码后的image 存入features中

  assert len(image.captions) == 1
  caption = image.captions[0]
  caption_ids = [vocab.word_to_id(word) for word in caption]#将描述的每个词对应的word转为id
  feature_lists = tf.train.FeatureLists(feature_list={
      "image/caption": _bytes_feature_list(caption),
      "image/caption_ids": _int64_feature_list(caption_ids)
  })
  sequence_example = tf.train.SequenceExample(
      context=context, feature_lists=feature_lists)
```

这里要注意的是，由于google原文是用的python2 在python3中由于文件读取的格式不太一样，这里要做一些改动

```python
def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    #（change）图片读取时已经是二进制
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _str2bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value,encoding="utf8")]))
```

然后将返回得到的sequence_example写入前面的写入流中就行了，生成tfrecord文件

```python
 writer.write(sequence_example.SerializeToString())
```



## 读取数据

> 这里开始就是模型的整体建立，首先是数据读取部分，如果是train和val模式的话就采用从tfrecord文件中用队列读取的方式进行快速的读取，如果是inference模式的话，就用placeholder进行单个数据的读取。这里主要说一下文件读取的方式。

生成tf.TFRecordReader()对象`reader=tf.TFRecordReader()`

生成文件名列表，根据传入的数据集地址

```python
data_files = []
for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
    
```

将文件名列表交给`tf.train.string_input_producer` 函数来生成一个先入先出的队列， 文件阅读器会需要它来读取数据。

```python
    #string_input_producer 提供的可配置参数来设置文件名乱序和最大的训练迭代数
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=True, capacity=16, name=shard_queue_name)
```

创建一个随机队列，最大长度为capacity，出队后的最小长度为min_after_dequeue

```python
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)
```

通过传入的tfrecordreader 对filename_queue队列进行阅读解码，返回的是一个example 样本

```python
  enqueue_ops = []
  for _ in range(num_reader_threads):
    _, value = reader.read(filename_queue)
    #将得到的value传入随机队列，并添加到enqueue_ops
    enqueue_ops.append(values_queue.enqueue([value]))
```

将队列加入到线程中

```python
#tf.train.queue_runner.QueueRunner（）持有一个队列的入列操作列表
  #queue 队列 enqueue_ops 用于线程中运行的入列操作列表
  #add_queue_runner 增加一个queue_runner到graph的收集器
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops))
```

将返回的队列进行dequeue()，得到序列化后的example，这里google用的也是多线程技术

```python
serialized_sequence_example = input_queue.dequeue()
```

对example进行解析，得到每个example的图片以及对应的描述

```python
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

```

对得到的图片进行预处理，预处理的方式用的就是inception的那一套，裁切啥啥的

对image和caption进行batch，因为我在后面的lstm部分想要加入attention机制，这时不能用dynamicRNN那个方法来实现（我没找到用dynamicRNN那个函数的实现attention的方法，那个函数方法可以允许batch内的序列长度不一致），而如果要根据num_step时间步长进行循环的话，要保证每个batch内，caption的长度要一致，不然会报错，这里我想到的一种方法是将较小的句子进行拼接，用一个循环实现，将每个句子都重复三次，（先将数据集中一些较短的句子剔除，比如小于四个单词的句子），这样每个句子都至少有16个单词那么长，然后再对所有的句子进行截断，截断长度16（可以自定义），然后再划分为输入语句和label语句，最后用tf.train.batch_join函数组成一个batch，并返回

```python
  enqueue_list = []
  for image, caption in images_and_captions:
    
    for _ in range(3):
      caption=tf.concat([caption,caption],0)
    
    caption=tf.slice(caption, [0], [num_step])
    
    caption_length = tf.shape(caption)[0]


    input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)#先减1再扩充维度
    #对caption进行切片操作，
    input_seq = tf.slice(caption, [0], input_length)#tf.slice(inputs,begin,size,name='')
    target_seq = tf.slice(caption, [1], input_length)
    indicator = tf.ones(input_length, dtype=tf.int64)
    enqueue_list.append([image, input_seq, target_seq, indicator])

  images, input_seqs, target_seqs, mask = tf.train.batch_join(
      enqueue_list,
      batch_size=batch_size,
      capacity=queue_capacity,
      dynamic_pad=True,
      name="batch_and_pad")
```



## 图片编码

> Image Caption 的整体框架就是一个encoede-decode过程，其中用CNN卷积网络对图片提取特征，相当于编码 

show and tell 论文里面CNN网络用的是inceptionv3，这里我想使用17年新出的densenet作为特征提取器，但由于个人资源有限，如果重新设计densenet的话训练起来比较困难，这里我使用的是github上找的一个densenet预训练模型，链接如下，使用的是其中的densenet161

> [https://github.com/pudae/tensorflow-densenet](https://github.com/pudae/tensorflow-densenet)

在show attend and tell 论文里面，使用的是VGG16提取的14×14×512的featuremaps，我这里改为densenet后，截取其中的14×14×384，某一个denseblock输出的featuremaps，再将featuremaps reshape一下，获得一个一个的向量196×384

```python
with slim.arg_scope(densenet.densenet_arg_scope()):
    #(inputs, num_classes=1000, data_format='NHWC', is_training=True, reuse=None):
    _, end_points = densenet.densenet161(images,is_training=is_densenet_model_training)
    with tf.variable_scope('densenet161'):
        net=end_points['densenet161/transition_block2']
        features=tf.reshape(net, [-1, net.shape[1]*net.shape[2], net.shape[3]])
```

这里我使用的方式是先restore全部的densenet weights，然后从end point 里面读取想要的那一层的net作为输出，在训练的时候，densenet部分的参数不参加训练

```python
#用densenet生成一个imageembedding
densenet_out = image_embedding.densenet_161(
    self.images,
    trainable=self.train_inception,
    is_training=self.is_training())
self.densenet_variable = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope="densenet161")

```



## word2vec

> 参考tensorflow的一个简单实现
>
> [https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)

embedding矩阵单独训练，根据token里面的描述，训练完后保存，再读入模型中

````python
seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)
````



## 模型主体构建

在最初的show and tell论文中，base model 在进行每一个词的预测时，没有考虑图片中相应的位置，接受的输入仅仅是上一步预测出的词以及上一步的隐藏层输出，图片的信息只在lstm初始化的时候有用到，这样子，图片的信息缺失其实是相当大的，而在show attend and tell 论文中，加入了图片中的位置信息，在每一步预测中，还会接受一个上下文的输入z。这样可以多增加一些图片的信息，避免在预测的时候一步出错，导致步步出错。

图片的编码上面已经讲过，经过编码后的特征向量用来初始化lstm的（c，h），论文中公式：
$$
c_0=f_{init,c}(\frac {1}{L} \sum_{i}^{L}{a_i})
$$

$$
h_0=f_{init,h}(\frac {1}{L} \sum_{i}^{L}{a_i})
$$

其中 f 表示一层全链接，代码实现如下：

```python
with tf.variable_scope('initial_lstm'):
    features_mean = tf.reduce_mean(features, 1)

    w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
    b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
    h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

    w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
    b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
    c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)

```

下面说一下如何生成context z，论文中分为两种形式，一种是hard attention 一种是soft attention 两种模式，这里我用了论文中soft的方式，而hard采用的是一种采样分布的方法，（具体的我没看明白。。。。），soft方式简单的来说就是对图片编码后的每个向量赋予一个权重比值，比值越高对下一个词的影响越大，比值的和为1，为了能够在每一步赋予a中的向量不同的权重，论文在CNN和lstm之间加入了一个多层感知机，根据论文中的公式对alpha进行计算：
$$
e_{ti}=f_{att}(a_i,h_{t-1})
$$
$$
\alpha_{ti}=\frac {exp(e_{ti})}{\sum {exp(e_{tk})}}
$$

上面的两道公式中，fatt我感知机，它接受的是图片向量本事，以及前一步lstm的输出h，这样子每一步对图片的不同位置的信息关注就和上一步输出有关，然后就可以得到一个上下文向量z：

$$
(\hat z_t)=\phi (\lbrace \alpha_{ti} \rbrace ,\lbrace a{i} \rbrace)
$$

其中的计算方式，直接根据alpha中的权重对a进行加权求和。代码实现如下：

```python
with tf.variable_scope('project_features'):
    w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
    features_flat = tf.reshape(features, [-1, self.D])
    features_proj = tf.matmul(features_flat, w)
    features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
    return features_proj
```

```python
with tf.variable_scope('attention_layer', reuse=reuse):
    w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
    b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
    w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

    h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b) 
    out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])
    alpha = tf.nn.softmax(out_att)
    context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')
```

首先经过一层感知机对图片信息经过一次处理，然后上一层的h也经过一个感知机，两者相加，再经过一层感知机，使用relu激活，然后对其变形，经过softmax得到alpha，再由alpha和原图特征向量加权求和得到context即论文中的z，在论文中说为了增强模型的效果，会对z乘上一个收缩因子即：
$$
\hat z_t=\phi (\lbrace \alpha_{ti} \rbrace ,\lbrace a{i} \rbrace)= \beta \sum {a_i \alpha_{ti}}
$$

$$
\beta_t=\sigma(f_{\beta}(h_{t-1}))
$$

代码实现如下：

```python
with tf.variable_scope('selector', reuse=reuse):
    w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
    b = tf.get_variable('b', [1], initializer=self.const_initializer)
    beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
    context = tf.multiply(beta, context, name='selected_context')
```

然后将编码后的词向量，以及上一层的h和生成的context输入lstm：

```python
with tf.variable_scope('lstm_cell', reuse=(t!=0)):
	_, (c, h) = lstm_cell(inputs=tf.concat( [x[:,t,:], context],1), state=[c, h])
```

根据论文中的描述，还需要根据词输入，h和context用一个较深的输出层对outputs进行计算，公式如下：
$$
p(y_t \mid a,y_1^{t-1})\infty exp(L_0(Ey_{t-1}+L_hh_t+L_z\hat z_t))
$$
其中的无穷符号表示啥意思我也不太清楚，按我的理解就是分别经过几个感知机后相加，最后再做一次全连接，代码如下：

```python
with tf.variable_scope('logits', reuse=reuse):
    w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
    b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)

    if dropout:
        h = tf.nn.dropout(h, 0.5)
        h_logits = tf.matmul(h, w_h) + b_h # self.M

    if self.ctx2out:
        w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M])
        h_logits += tf.matmul(context, w_ctx2out)#self.M

    if self.prev2out:
		h_logits += x
        h_logits = tf.nn.tanh(h_logits)

    if dropout:
        h_logits = tf.nn.dropout(h_logits, 0.5)
 
```

上面的过程还要再循环num_step次，形成RNN的循环网络，将每一次的输出的h_logits进行存储，最后经过一个全连接得到output_logits，分类数为文本前面统计的单词个数。

在论文中还提到，最好让alpha在时间维度上也能保证加和等于1，这样有利于模型效果，为了实现以它，可以在loss进行一下修改：
$$
L_d=-log(P(y\mid x))+\lambda \sum_i^L{(1-\sum_t^C{\alpha_{ti}})^2}
$$
代码实现如下;

```python
alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))
alphas_all = tf.reduce_sum(alphas, 1)
alpha_reg = self.alpha_c * tf.reduce_sum((1 - alphas_all) ** 2)
```

最后loss采用交叉熵损失

```python
targets = tf.reshape(captions_out, [-1])
weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

# Compute losses.
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,logits=logits)+alpha_reg
batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                    tf.reduce_sum(weights),
                    name="batch_loss")
tf.losses.add_loss(batch_loss)
```

到此，模型主体基本上构建完毕



## 模型训练

> 模型构建完毕后，需要构建优化器等，（在im2txt中令我感到困惑的地方，我始终找不到跟输出文件夹联系的地方）

**初始化CNN**

由于CNN使用的是在imagnet上预训练过的densenet，所以在模型加载的时候需要读取这一部分的权重

```python
saver = tf.train.Saver(self.densenet_variable)
def restore_fn(sess):
    tf.logging.info("Restoring Inception variables from checkpoint file %s",
                    self.config.inception_checkpoint_file)
    saver.restore(sess, self.config.inception_checkpoint_file)
```

 **设置 global step**

```python
global_step = tf.Variable(
    initial_value=0,
    name="global_step",
    trainable=False,
    collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
```
**设置learning rate**

```python
learning_rate = tf.constant(FLAGS.learning_rate)
if training_config.learning_rate_decay_factor > 0:
    num_batches_per_epoch = (training_config.num_examples_per_epoch /
                             model_config.batch_size)
    decay_steps = int(num_batches_per_epoch *
                      training_config.num_epochs_per_decay)

    def _learning_rate_decay_fn(learning_rate, global_step):
        return tf.train.exponential_decay(
            learning_rate,
            global_step,
            decay_steps=decay_steps,
            decay_rate=training_config.learning_rate_decay_factor,
            staircase=True)
```
**收集要训练的变量以及训练会话**

这里我想要冻结CNN部分的参数，不加入参数优化，仅作为特征提取器用，感觉如果训练的话，会很困难....但是这里使用tensorflow官方的im2txt里面的参数配置好像不能冻结densenet的权重（可能是我另外读取的原因），还是会被训练，所以这里我收集需要被优化的变量，加入run操作中

```python
output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'lstm')
```

```python
    train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss,
        global_step=model.global_step,
        learning_rate=learning_rate,
        optimizer=training_config.optimizer,
        clip_gradients=training_config.clip_gradients,
        learning_rate_decay_fn=learning_rate_decay_fn,
        variables=output_vars)
```



**训练结果**

![](https://i.imgur.com/E1BcURZ.png)


## 模型校验

大致步骤和一般的模型一样，先建立模型，然后restore ckpt的权重，再run一次整个模型，得到softmax 后的logits，再对比label，用评价指标进行校验，这里使用的是tensorflow里面的校验脚本

**建立模型**

```python
model_config = configuration.ModelConfig()
model_config.input_file_pattern = FLAGS.input_file_pattern
model = show_attend_and_tell.ShowAndTellModel(model_config, mode="eval")
model.build()
```

**读取权重**

```python
model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
saver.restore(sess, model_path)
```

**开始数据队列**

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
**计算损失**

```python
  # Compute perplexity over the entire dataset.
  num_eval_batches = int(
      math.ceil(FLAGS.num_eval_examples / model.config.batch_size))

  start_time = time.time()
  sum_losses = 0.
  sum_weights = 0.
  for i in range(num_eval_batches):
    cross_entropy_losses, weights = sess.run([
        model.target_cross_entropy_losses,
        model.target_cross_entropy_loss_weights
    ])
    sum_losses += np.sum(cross_entropy_losses * weights)
    sum_weights += np.sum(weights)
    if not i % 100:
      tf.logging.info("Computed losses for %d of %d batches.", i + 1,
                      num_eval_batches)
  eval_time = time.time() - start_time

  perplexity = math.exp(sum_losses / sum_weights)
```



## 模型inference

也是参照的tensorflow im2txt的inference，核心算法是里面的beam search 算法

**创建graph**

```python
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()
```

```python
  def build_model(self, model_config):
    model = show_attend_and_tell.ShowAndTellModel(model_config, mode="inference")
    model.build()
    return model
```



**从节点中恢复参数**

```python
with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)
```

**caption生成器**

创建一个caption生成器实例，里面包含了beam search 算法

```python
generator = caption_generator.CaptionGenerator(model, vocab)
```

**beam search**

beam search 的核心思想就是设置一个k，每次预测的时候取前k个词，和原来的已经生成的k个句子进行组合，形成k×k个句子，然后对这k×k个句子进行评价打分，用的是对数概率累加好像，抽出前k个评分最高的句子，并将新预测的词的相关状态等再返回模型进行新一轮的预测，重复这个过程，其中保存前k个语句的时候用到了堆排序的方法，具体的代码实现如下

* 根据图片得到初始的lstm state

  ```python
  initial_state = self.model.feed_image(sess, encoded_image)
  ```

* 生成一个beam Caption实例

  一个Caption实例存储了一个当前的sentence语句，当前的state状态，当前语句的logprob对数几率，当前语句的评分score，定义如下：

  ```python
  class Caption(object):
    """Represents a complete or partial caption."""
  
    def __init__(self, sentence, state, logprob, score, metadata=None):
      """Initializes the Caption.
  
      Args:
        sentence: List of word ids in the caption.
        state: Model state after generating the previous word.
        logprob: Log-probability of the caption.
        score: Score of the caption.
        metadata: Optional metadata associated with the partial sentence. If not
          None, a list of strings with the same length as 'sentence'.
      """
      self.sentence = sentence
      self.state = state
      self.logprob = logprob
      self.score = score
      self.metadata = metadata
  
    def __cmp__(self, other):
      """Compares Captions by score."""
      assert isinstance(other, Caption)
      if self.score == other.score:
        return 0
      elif self.score < other.score:
        return -1
      else:
        return 1
    
    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
      assert isinstance(other, Caption)
      return self.score < other.score
    
    # Also for Python 3 compatibility.
    def __eq__(self, other):
      assert isinstance(other, Caption)
      return self.score == other.score
  ```

  将初始状态生成一个Caption，并为当前语句赋一个开始字符，用于开始预测

  ```python
  initial_beam = Caption(
      sentence=[self.vocab.start_id],
      state=initial_state[0],
      logprob=0.0,
      score=0.0,
      metadata=[""])
  ```

* 生成TopN实例，用以储存Caption对象，并根据对象的score将其排序，其中只保存前k个Caption实例，相当于是对生成的预测语句进行筛选，一个TopN实例定义如下

  ```python
  class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""
  
    def __init__(self, n):
      self._n = n
      self._data = []
  
    def size(self):
      assert self._data is not None
      return len(self._data)
  
    def push(self, x):
      """Pushes a new element."""
      assert self._data is not None
      # n = beamsize
      if len(self._data) < self._n:
        heapq.heappush(self._data, x)# 把一项值压入堆heap，同时维持堆的排序要求。小的在上面
      else:
        heapq.heappushpop(self._data, x)# 向堆里插入一项，并返回最小值的项。组合了前面两个函数，这样更加有效率
  
    def extract(self, sort=False):
      
      assert self._data is not None
      data = self._data
      self._data = None
      if sort:
        data.sort(reverse=True)
      return data
  
    def reset(self):
      """Returns the TopN to an empty state."""
      self._data = []
  ```

  生成一个临时caption储存器以及一个终极的caption存储器，临时的用来循环的时候不断筛选sentence，终极的用来生成最后的输出，将初始的beam存入临时储存器中

  ```python
  partial_captions = TopN(self.beam_size)
  partial_captions.push(initial_beam)
  complete_captions = TopN(self.beam_size)
  ```

* Run beam search

  在最大允许语句长度内循环beam search，在每一次循环中，先将临时存储器中的caption实例取出，并将存储器清空

  ```python
  partial_captions_list = partial_captions.extract()
  partial_captions.reset()
  ```

  将上一个预测的词作为词的输入，上一个状态作为状态的输入，输入模型后返回得到的softmax结果，新的lstm状态

  ```python
  input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
  state_feed = np.array([c.state for c in partial_captions_list]) 
  softmax, new_states, metadata = self.model.inference_step(sess,
                                                            input_feed,
                                                            state_feed,
                                                            encoded_image)
  ```

  对提取出的caotion实例进行遍历，根据得到的softmax结果取概率最大的前k个，并将这k个和原来的句子进行组合，得到k个sentence，并将所得对数概率和句子的对数概率相加，作为新句子的score，如果新预测的词是结束符的话，将这个caption实例存入终极存储器，不是的话存入临时存储器，并继续下一个caption的操作，每个存入临时存储器的caption都会根据自身的score和其他caption的score进行对比，存储器中只保留前k个caption实例，直到caption遍历完后，进行下一次预测

  ```python
        for i, partial_caption in enumerate(partial_captions_list):
          #按batch顺序循环，得到softmax，state
          word_probabilities = softmax[i]
          state = new_states[i]
          # For this partial caption, get the beam_size most probable next words.
          #将softmax的值和所在索引 转为list
          words_and_probs = list(enumerate(word_probabilities))
          words_and_probs.sort(key=lambda x: -x[1])#进行排序
          words_and_probs = words_and_probs[0:self.beam_size]#取前三个
          # Each next word gives a new partial caption.
          for w, p in words_and_probs:
            if p < 1e-12:
              continue  # Avoid log(0).
            sentence = partial_caption.sentence + [w]
  
            logprob = partial_caption.logprob + math.log(p)
            print(w,p,sentence,logprob)
            score = logprob
            if metadata:
              metadata_list = partial_caption.metadata + [metadata[i]]
            else:
              metadata_list = None
            # 如果这个词是结束符的话，输出
            if w == self.vocab.end_id:
              if self.length_normalization_factor > 0:
                score /= len(sentence)**self.length_normalization_factor
              beam = Caption(sentence, state, logprob, score, metadata_list)
              complete_captions.push(beam)
              print('end')
            else:
              beam = Caption(sentence, state, logprob, score, metadata_list)
              partial_captions.push(beam)
  ```

  如果循环结束后，还没有预测出结束符，即终极存储器是空的，就将临时存储器中的caption作为结果返回

* 输出语句

  将beam search返回的caption实例中的sentence提取出来，将word id 根据 vocabulary转为word，打印出结果

  ```python
  for i, caption in enumerate(captions):
      # Ignore begin and end words.
      sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
      sentence = " ".join(sentence)
      print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
  ```

**预测结果**

对训练数据的预测

![](https://i.imgur.com/xr8E6gb.jpg)
## 模型移植



















----

**参考文章等**

[https://blog.csdn.net/shenxiaolu1984/article/details/51493673#fnref:1](https://blog.csdn.net/shenxiaolu1984/article/details/51493673#fnref:1)

[https://github.com/tensorflow/models/tree/master/research/im2txt](https://github.com/tensorflow/models/tree/master/research/im2txt)

[https://blog.csdn.net/chazhongxinbitc/article/details/78689754](https://blog.csdn.net/chazhongxinbitc/article/details/78689754)

[https://blog.csdn.net/luoyang224/article/details/76599736](https://blog.csdn.net/luoyang224/article/details/76599736)





