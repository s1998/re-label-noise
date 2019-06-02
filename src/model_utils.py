import tensorflow as tf
import numpy as np

max_length = 120
batch_size = 64


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def word_embedding(word, word_vec_mat, var_scope=None, word_embedding_dim=50, add_unk_start_end_blank=True):
    with tf.variable_scope(var_scope or 'word_embedding', reuse=tf.AUTO_REUSE):
        word_embedding = tf.get_variable('word_embedding', initializer=word_vec_mat.astype(np.float32), dtype=tf.float32)
        if add_unk_start_end_blank:
            word_embedding = tf.concat([word_embedding,
                                        tf.get_variable("unk_word_embedding", [1, word_embedding_dim], dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32)),
                                        tf.get_variable("start_word_embedding", [1, word_embedding_dim], dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32)),
                                        tf.get_variable("end_word_embedding", [1, word_embedding_dim], dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32)),
                                        tf.constant(np.zeros((1, word_embedding_dim), dtype=np.float32))], 0)
        x = tf.nn.embedding_lookup(word_embedding, word)
        return x

def pos_embedding(pos1, pos2, var_scope=None, pos_embedding_dim=5, max_length=130):
    with tf.variable_scope(var_scope or 'pos_embedding', reuse=tf.AUTO_REUSE):
        pos_tot = max_length * 2

        pos1_embedding = tf.get_variable('real_pos1_embedding', [pos_tot, pos_embedding_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32)) 
        pos2_embedding = tf.get_variable('real_pos2_embedding', [pos_tot, pos_embedding_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32)) 

        input_pos1 = tf.nn.embedding_lookup(pos1_embedding, pos1)
        input_pos2 = tf.nn.embedding_lookup(pos2_embedding, pos2)
        x = tf.concat([input_pos1, input_pos2], -1)
        return x

def word_position_embedding(word, word_vec_mat, pos1, pos2, var_scope=None, word_embedding_dim=50, pos_embedding_dim=5, max_length=120, add_unk_start_end_blank=True):
    w_embedding = word_embedding(word, word_vec_mat, var_scope=var_scope, word_embedding_dim=word_embedding_dim, add_unk_start_end_blank=add_unk_start_end_blank)
    p_embedding = pos_embedding(pos1, pos2, var_scope=var_scope, pos_embedding_dim=pos_embedding_dim, max_length=max_length)
    return tf.concat([w_embedding, p_embedding], -1)

def __dropout__(x, keep_prob=1.0):
    return tf.contrib.layers.dropout(x, keep_prob=keep_prob)

def __piecewise_pooling__(x, mask):
    mask_embedding = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    mask = tf.nn.embedding_lookup(mask_embedding, mask)
    hidden_size = x.shape[-1]
    # print("59 : ", x.shape, hidden_size, mask.shape)
    # print("60 : ", tf.expand_dims(mask * 100, 2).shape, tf.expand_dims(x, 3).shape)
    x = tf.reduce_max(tf.expand_dims(mask * 100, 2) + tf.expand_dims(x, 3), axis=1) - 100
    return tf.reshape(x, [-1, hidden_size * 3])

def __cnn_cell__(x, hidden_size=230, kernel_size=3, stride_size=1):
    x = tf.layers.conv1d(inputs=x, 
                         filters=hidden_size, 
                         kernel_size=kernel_size, 
                         strides=stride_size, 
                         padding='same', 
                         kernel_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    return x

def __rnn_cell__(hidden_size, cell_name='lstm'):
    if isinstance(cell_name, list) or isinstance(cell_name, tuple):
        if len(cell_name) == 1:
            return __rnn_cell__(hidden_size, cell_name[0])
        cells = [self.__rnn_cell__(hidden_size, c) for c in cell_name]
        return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    if cell_name.lower() == 'lstm':
        return tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True, activation=tf.nn.tanh)
    elif cell_name.lower() == 'gru':
        print("Using gru cell")
        return tf.contrib.rnn.GRUCell(hidden_size)

def crnn(x, length, mask, hidden_size=230, cell_name='gru', var_scope=None, keep_prob=1.0, kernel_size=3, stride_size=1):
    with tf.variable_scope(var_scope or "rnn", reuse=tf.AUTO_REUSE):
        # x = __dropout__(x, keep_prob)
        print("74 : ", x.shape)
        x = __cnn_cell__(x, 50, kernel_size, stride_size)
        cell = tf.contrib.rnn.DropoutWrapper(__rnn_cell__(hidden_size, cell_name), output_keep_prob=keep_prob)
        outputs, states = tf.nn.dynamic_rnn(
          cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic-rnn')
        print("78 : ", outputs.shape)
        # return tf.reshape(outputs[: ,-1, :], [-1, hidden_size]) 
        return states 

def crnn2(x, length, mask, hidden_size=230, cell_name='gru', var_scope=None, keep_prob=1.0, kernel_size=3, stride_size=1):
    with tf.variable_scope(var_scope or "rnn", reuse=tf.AUTO_REUSE):
        # x = __dropout__(x, keep_prob)
        print("74 : ", x.shape)
        x = __cnn_cell__(x, 50, kernel_size, stride_size)
        x = tf.nn.relu(x)
        cell = tf.contrib.rnn.DropoutWrapper(__rnn_cell__(hidden_size, cell_name), output_keep_prob=keep_prob)
        outputs, states = tf.nn.dynamic_rnn(
          cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic-rnn')
        print("78 : ", outputs.shape)
        # return tf.reshape(outputs[: ,-1, :], [-1, hidden_size]) 
        return states 

def rnn(x, length, mask, hidden_size=230, cell_name='gru', var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "rnn", reuse=tf.AUTO_REUSE):
        # x = __dropout__(x, keep_prob)
        print("74 : ", x.shape)
        cell = tf.contrib.rnn.DropoutWrapper(__rnn_cell__(hidden_size, cell_name), output_keep_prob=keep_prob)
        outputs, states = tf.nn.dynamic_rnn(
          cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic-rnn')
        print("78 : ", outputs.shape)
        # return tf.reshape(outputs[: ,-1, :], [-1, hidden_size]) 
        return states 

def bgwa(x, length, mask, hidden_size=192, cell_name="gru", var_scope=None, keep_prob=1.0):
  with tf.variable_scope(var_scope or "rnn", reuse=tf.AUTO_REUSE):
    fw_cell = tf.contrib.rnn.DropoutWrapper(__rnn_cell__(hidden_size, cell_name), output_keep_prob=keep_prob)
    bw_cell = tf.contrib.rnn.DropoutWrapper(__rnn_cell__(hidden_size, cell_name), output_keep_prob=keep_prob)
    out, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic-bi-rnn')
    x = tf.concat((out[0], out[1]), axis=2)
    wrd_query = tf.get_variable('wrd_query', [2*hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())
    word_att = tf.nn.softmax(tf.reshape( tf.matmul(tf.reshape(tf.tanh(x),[-1, 2*hidden_size]), wrd_query), [-1, max_length])),
    word_att = tf.reshape(word_att, [-1, 1, max_length])
    x  = tf.reshape(tf.matmul(word_att,x), [-1, 2*hidden_size])
    return x

def pbrnn(x, length, mask, hidden_size=100, cell_name="gru", var_scope=None, keep_prob=1.0):
  with tf.variable_scope(var_scope or "rnn", reuse=tf.AUTO_REUSE, initializer = tf.initializers.truncated_normal(0.0, 0.1)):
    fw_cell = tf.contrib.rnn.DropoutWrapper(__rnn_cell__(hidden_size, cell_name), output_keep_prob=keep_prob)
    bw_cell = tf.contrib.rnn.DropoutWrapper(__rnn_cell__(hidden_size, cell_name), output_keep_prob=keep_prob)
    out, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic-bi-rnn')
    x = tf.concat((out[0], out[1]), axis=2)
    x = __piecewise_pooling__(x, mask)
    x = __dropout__(x, keep_prob)
    return x

def rnn2(x, length, mask, hidden_size=230, cell_name='gru', var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "rnn", reuse=tf.AUTO_REUSE):
        # x = __dropout__(x, keep_prob)
        print("74 : ", x.shape)
        cell = tf.contrib.rnn.DropoutWrapper(__rnn_cell__(hidden_size, cell_name), output_keep_prob=keep_prob)
        outputs, states = tf.nn.dynamic_rnn(
          cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic-rnn')
        print("78 : ", outputs.shape)
        # return tf.reshape(outputs[: ,-1, :], [-1, hidden_size]) 
        return tf.nn.relu(states )

def birnn(x, length, mask, hidden_size=230, cell_name='gru', var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "birnn", reuse=tf.AUTO_REUSE):
        fw_cell = tf.contrib.rnn.DropoutWrapper(__rnn_cell__(hidden_size, cell_name), output_keep_prob=keep_prob)
        bw_cell = tf.contrib.rnn.DropoutWrapper(__rnn_cell__(hidden_size, cell_name), output_keep_prob=keep_prob)
        _, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic-bi-rnn')
        fw_states, bw_states = states
        if isinstance(fw_states, tuple):
            fw_states = fw_states[0]
            bw_states = bw_states[0]
        return tf.concat([fw_states, bw_states], axis=1)

def birnn2(x, length, mask, hidden_size=230, cell_name='gru', var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "birnn", reuse=tf.AUTO_REUSE):
        fw_cell = tf.contrib.rnn.DropoutWrapper(__rnn_cell__(hidden_size, cell_name), output_keep_prob=keep_prob)
        bw_cell = tf.contrib.rnn.DropoutWrapper(__rnn_cell__(hidden_size, cell_name), output_keep_prob=keep_prob)
        _, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic-bi-rnn')
        fw_states, bw_states = states
        if isinstance(fw_states, tuple):
            fw_states = fw_states[0]
            bw_states = bw_states[0]
        return tf.nn.relu(tf.concat([fw_states, bw_states], axis=1))

def pcnn(x, length, mask, hidden_size=230 , kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "pcnn", reuse=tf.AUTO_REUSE):
        max_length = x.shape[1]
        x = __cnn_cell__(x, hidden_size, kernel_size, stride_size)
        print("121", x.shape)
        x = __piecewise_pooling__(x, mask)
        x = activation(x)
        x = __dropout__(x, keep_prob)
        return x

def pcnn2tl(x, length, mask, hidden_size=230 , kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    max_length = x.shape[1]
    with tf.variable_scope(var_scope or "pcnn", reuse=tf.AUTO_REUSE):
      with tf.variable_scope("l1", reuse=tf.AUTO_REUSE):
        x = __cnn_cell__(x, 70, kernel_size, stride_size)
        x = activation(x)
      with tf.variable_scope("l2", reuse=tf.AUTO_REUSE):
        x = __cnn_cell__(x, 80, kernel_size, stride_size)
        x = activation(x)
      with tf.variable_scope("l3", reuse=tf.AUTO_REUSE):
        x = __cnn_cell__(x, 80, kernel_size, stride_size)
        x = __piecewise_pooling__(x, mask)
        x = activation(x)
        x = __dropout__(x, keep_prob)
        return x

def pcnn2(x, length, mask, hidden_size=230 , kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "pcnn", reuse=tf.AUTO_REUSE):
      max_length = x.shape[1]
      x = __cnn_cell__(x, 70, kernel_size, stride_size)
      x = activation(x)
      x = __cnn_cell__(x, 80, kernel_size, stride_size)
      x = activation(x)
      x = __cnn_cell__(x, 80, kernel_size, stride_size)
      x = __piecewise_pooling__(x, mask)
      x = activation(x)
      x = __dropout__(x, keep_prob)
      return x

def pcnn2n(x, length, mask, hidden_size=230 , kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    print("here")
    with tf.variable_scope(var_scope or "pcnn", reuse=tf.AUTO_REUSE):
      max_length = x.shape[1]
      with tf.variable_scope("layer1", reuse=tf.AUTO_REUSE):
        x = __cnn_cell__(x, 70, kernel_size, stride_size)
        x = activation(x)
      with tf.variable_scope("layer2", reuse=tf.AUTO_REUSE):
        x = __cnn_cell__(x, 80, kernel_size, stride_size)
        x = activation(x)
      with tf.variable_scope("layer3", reuse=tf.AUTO_REUSE):
        x = __cnn_cell__(x, 80, kernel_size, stride_size)
      x = __piecewise_pooling__(x, mask)
      x = activation(x)
      x = __dropout__(x, keep_prob)
      return x

def __logit__(x, rel_tot, var_scope=None):
    with tf.variable_scope(var_scope or 'logit', reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', shape=[rel_tot], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        logit = tf.matmul(x, tf.transpose(relation_matrix)) + bias
    return logit

def __attention_train_logit__(x, query, rel_tot, var_scope=None):
    with tf.variable_scope(var_scope or 'logit', reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', shape=[rel_tot], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    current_relation = tf.nn.embedding_lookup(relation_matrix, query)
    attention_logit = tf.reduce_sum(current_relation * x, -1) # sum[(n', hidden_size) \dot (n', hidden_size)] = (n)
    return attention_logit

def __attention_test_logit__(x, rel_tot, var_scope=None):
    with tf.variable_scope(var_scope or 'logit', reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', shape=[rel_tot], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    attention_logit = tf.matmul(x, tf.transpose(relation_matrix)) # (n', hidden_size) x (hidden_size, rel_tot) = (n', rel_tot)
    return attention_logit

def bag_attention(x, scope, query, rel_tot, is_training, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "attention", reuse=tf.AUTO_REUSE):
        if is_training: # training
            bag_repre = []
            attention_logit = __attention_train_logit__(x, query, rel_tot)
            att_scores = []
            for i in range(scope.shape[0]):
                bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                attention_score = tf.nn.softmax(attention_logit[scope[i][0]:scope[i][1]], -1)
                bag_repre.append(tf.squeeze(tf.matmul(tf.expand_dims(attention_score, 0), bag_hidden_mat))) # (1, n') x (n', hidden_size) = (1, hidden_size) -> (hidden_size)
                att_scores.append(attention_score)
            bag_repre = tf.stack(bag_repre)
            bag_repre = __dropout__(bag_repre, keep_prob)
            return __logit__(bag_repre, rel_tot), bag_repre, att_scores
        else: # testing
            attention_logit = __attention_test_logit__(x, rel_tot) # (n, rel_tot)
            bag_repre = [] 
            bag_logit = []
            att_scores = []
            for i in range(scope.shape[0]):
                bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                attention_score = tf.nn.softmax(tf.transpose(attention_logit[scope[i][0]:scope[i][1], :]), -1) # softmax of (rel_tot, n')
                att_scores.append(attention_score)
                bag_repre_for_each_rel = tf.matmul(attention_score, bag_hidden_mat) # (rel_tot, n') \dot (n', hidden_size) = (rel_tot, hidden_size)
                bag_logit_for_each_rel = __logit__(bag_repre_for_each_rel, rel_tot) # -> (rel_tot, rel_tot)
                bag_repre.append(bag_repre_for_each_rel)
                bag_logit.append(tf.diag_part(tf.nn.softmax(bag_logit_for_each_rel, -1))) 
            bag_repre = tf.stack(bag_repre)
            bag_logit = tf.stack(bag_logit)
            return bag_logit, bag_repre, att_scores

def softmax_cross_entropy(x, label, rel_tot, weights_table=None, var_scope=None):
    with tf.variable_scope(var_scope or "loss", reuse=tf.AUTO_REUSE):
        if weights_table is None:
            weights = 1.0
        else:
            weights = tf.nn.embedding_lookup(weights_table, label)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=x, weights=weights)
        tf.summary.scalar('loss', loss)
        return loss

def bag_cross_max(x, scope, rel_tot, var_scope=None, dropout_before=False, keep_prob=1.0):
  '''
  Cross-sentence Max-pooling proposed by (Jiang et al. 2016.)
  "Relation Extraction with Multi-instance Multi-label Convolutional Neural Networks"
  https://pdfs.semanticscholar.org/8731/369a707046f3f8dd463d1fd107de31d40a24.pdf
  '''
  with tf.variable_scope("cross_max", reuse=tf.AUTO_REUSE):
      if dropout_before:
          x = __dropout__(x, keep_prob)
      bag_repre = []
      for i in range(scope.shape[0]):
          bag_hidden_mat = x[scope[i][0]:scope[i][1]]
          bag_repre.append(tf.reduce_max(bag_hidden_mat, 0)) # (n', hidden_size) -> (hidden_size)
      bag_repre = tf.stack(bag_repre)
      # if not dropout_before:
      #     bag_repre = __dropout__(bag_repre, keep_prob)
  att_scores = tf.constant(0)
  return __logit__(bag_repre, rel_tot), bag_repre, att_scores


all_encoders = {"pcnn" : pcnn, "pbrnn" : pbrnn, "pcnn2" : pcnn2, "pcnn2n" : pcnn2n, 
				"pcnn2n" : pcnn2n, "rnn" : rnn, "brnn" : brnn, "rnn2" : rnn2, "brnn2" : brnn2, 
				"crnn" : crnn, "crnn2" : crnn2, "bgwa" : bgwa}

def loss_none(train_logit, labels, rel_tot, bs_val = 0.0, l2_lamd = 0.0):
    loss = softmax_cross_entropy(train_logit, 
            tf.one_hot(labels, rel_tot), rel_tot)
	print("Created model with no bootstrapping, bs val : {}".format(bs_val))
    return loss

def loss_hard(train_logit, labels, rel_tot, bs_val = 0.0, l2_lamd = 0.0):
	selected_pos_labels = tf.reduce_max(train_logit[:, 1:], axis = 1) > 0.95
	train_labels_new = tf.where(selected_pos_labels, 
	  tf.argmax(train_logit[:, 1:], axis = 1) + 1,
	  labels)
	train_labels_new = tf.stop_gradient(train_labels_new)
	train_labels_new = tf.cast(tf.one_hot(train_labels_new, rel_tot), dtype=tf.float32) * (1 - bs_val) + \
		bs_val * tf.cast(tf.one_hot(labels, rel_tot), dtype=tf.float32)
	loss = softmax_cross_entropy(train_logit, train_labels_new, rel_tot)
	print("Created model with hard bootstrapping, bs val : {}".format(bs_val))
	return loss

def loss_soft(train_logit, labels, rel_tot, bs_val = 0.0, l2_lamd = 0.0):
	train_labels_ = tf.cast(train_logit, dtype=tf.float32) * (1 - bs_val) + \
            (bs_val) * tf.cast(tf.one_hot(labels, rel_tot), dtype=tf.float32)
	loss = softmax_cross_entropy(train_logit, train_labels_, rel_tot)
	print("Created model with soft bootstrapping, bs val : {}".format(bs_val))
    return loss

def loss_extra(train_logit, labels, rel_tot, bs_val = 0.0, l2_lamd = 0.0):
	extra_layer = 0.98 * tf.Variable(tf.eye(rel_tot), name = "extra_layer_identity") + \
							tf.get_variable("extra_layer_noise", [rel_tot, rel_tot], dtype=tf.float32,
	                                initializer=tf.initializers.truncated_normal(0, 0.01, dtype=tf.float32))
	new_logits = tf.matmul(train_logit, extra_layer)
	loss = softmax_cross_entropy(new_logits, 
								tf.one_hot(labels, rel_tot), rel_tot) + l2_lambda * tf.nn.l2_loss(extra_layer)
	print("Created model with extra layer")
	return loss

all_losses = {"none" : loss_none, "extra" : loss_extra, "hard" : loss_hard, "soft" : loss_soft}