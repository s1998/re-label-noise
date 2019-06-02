from src.utils import load_word_vec
import tensorflow as tf
import numpy as np

max_length = 128
batch_size = 128
no_of_classes = 503

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

def pos_embedding(pos1, pos2, var_scope=None, pos_embedding_dim=5, max_length=120):
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

def __dropout__(x, keep_prob):
    return tf.contrib.layers.dropout(x, keep_prob=keep_prob)

def __piecewise_pooling__(x, mask):
    mask_embedding = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    mask = tf.nn.embedding_lookup(mask_embedding, mask)
    hidden_size = x.shape[-1]
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
        return tf.contrib.rnn.GRUCell(hidden_size, activation=tf.nn.tanh)

def rnn(x, length, hidden_size=230, cell_name='lstm', var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "rnn", reuse=tf.AUTO_REUSE):
        x = __dropout__(x, keep_prob)
        print("74 : ", x.shape)
        cell = __rnn_cell__(hidden_size, cell_name)
        outputs, states = tf.nn.dynamic_rnn(
          cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic-rnn')
        print("78 : ", outputs.shape)
        return tf.reshape(outputs[: ,-1, :], [-1, hidden_size]) 

def pcnn(x, mask, keep_prob, hidden_size=230 , kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None):
    with tf.variable_scope(var_scope or "pcnn", reuse=tf.AUTO_REUSE):
        max_length = x.shape[1]
        x = __cnn_cell__(x, hidden_size, kernel_size, stride_size)
        x = __piecewise_pooling__(x, mask)
        x = activation(x)
        x = __dropout__(x, keep_prob)
        return x

def __logit__(x, rel_tot, var_scope=None):
    with tf.variable_scope(var_scope or 'logit', reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        bias = tf.get_variable('bias', shape=[rel_tot], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        logit = tf.matmul(x, tf.transpose(relation_matrix)) + bias
    return logit

def __attention_train_logit__(x, query, rel_tot, var_scope=None):
    with tf.variable_scope(var_scope or 'logit', reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        bias = tf.get_variable('bias', shape=[rel_tot], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    current_relation = tf.nn.embedding_lookup(relation_matrix, query)
    attention_logit = tf.reduce_sum(current_relation * x, -1) # sum[(n', hidden_size) \dot (n', hidden_size)] = (n)
    return attention_logit

def __attention_test_logit__(x, rel_tot, var_scope=None):
    with tf.variable_scope(var_scope or 'logit', reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        bias = tf.get_variable('bias', shape=[rel_tot], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    attention_logit = tf.matmul(x, tf.transpose(relation_matrix)) # (n', hidden_size) x (hidden_size, rel_tot) = (n', rel_tot)
    return attention_logit

def bag_attention(x, scope, query, rel_tot, is_training, keep_prob, var_scope=None):
    with tf.variable_scope(var_scope or "attention", reuse=tf.AUTO_REUSE):
        if is_training: # training
            bag_repre = []
            attention_logit = __attention_train_logit__(x, query, rel_tot)
            for i in range(scope.shape[0]):
                bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                attention_score = tf.nn.softmax(attention_logit[scope[i][0]:scope[i][1]], -1)
                bag_repre.append(tf.squeeze(tf.matmul(tf.expand_dims(attention_score, 0), bag_hidden_mat))) # (1, n') x (n', hidden_size) = (1, hidden_size) -> (hidden_size)
            bag_repre = tf.stack(bag_repre)
            bag_repre = __dropout__(bag_repre, keep_prob)
            return __logit__(bag_repre, rel_tot), bag_repre
        else: # testing
            attention_logit = __attention_test_logit__(x, rel_tot) # (n, rel_tot)
            bag_repre = [] 
            bag_logit = []
            for i in range(scope.shape[0]):
                bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                attention_score = tf.nn.softmax(tf.transpose(attention_logit[scope[i][0]:scope[i][1], :]), -1) # softmax of (rel_tot, n')
                bag_repre_for_each_rel = tf.matmul(attention_score, bag_hidden_mat) # (rel_tot, n') \dot (n', hidden_size) = (rel_tot, hidden_size)
                bag_logit_for_each_rel = __logit__(bag_repre_for_each_rel, rel_tot) # -> (rel_tot, rel_tot)
                bag_repre.append(bag_repre_for_each_rel)
                bag_logit.append(tf.diag_part(tf.nn.softmax(bag_logit_for_each_rel, -1))) # could be improved by sigmoid?
            bag_repre = tf.stack(bag_repre)
            bag_logit = tf.stack(bag_logit)
            return bag_logit, bag_repre

def softmax_cross_entropy(x, label, rel_tot, weights_table=None, var_scope=None):
    with tf.variable_scope(var_scope or "loss", reuse=tf.AUTO_REUSE):
        if weights_table is None:
            weights = 1.0
        else:
            weights = tf.nn.embedding_lookup(weights_table, label)
        label_onehot = tf.one_hot(indices=label, depth=rel_tot, dtype=tf.int32)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=label_onehot, logits=x, weights=weights)
        tf.summary.scalar('loss', loss)
        return loss

class Model:
    def __init__(self,
        word_vec_mat,
        encoder = "pcnn", selector="att", no_of_classes = 503): 
        print("Creating model with encoder and selector : ", encoder, selector)
        self.encoder = encoder
        self.selector = selector
        self.words_ph = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='word')
        self.pos1_ph = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos1')
        self.pos2_ph = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos2')
        self.labels_ph = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='label')
        self.ins_labels_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='ins_label')
        self.lengths_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='length')
        self.scope_ph = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2], name='scope')
        self.masks_ph = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="mask")
        self.rel_tot = no_of_classes
        self.keep_prob = tf.placeholder(tf.float32)

        self.x = word_position_embedding(
            self.words_ph, word_vec_mat, self.pos1_ph, self.pos2_ph)

        if encoder == "pcnn":
            self.sent_enc = pcnn(self.x, self.masks_ph, keep_prob=self.keep_prob)
        elif encoder == "rnn":
            self.sent_enc = rnn(self.x, self.lengths_ph, keep_prob=self.keep_prob)
        print(self.sent_enc.shape)

        self.train_logit, self.train_repre = bag_attention(
            self.sent_enc, self.scope_ph, 
            self.ins_labels_ph, self.rel_tot, True, keep_prob=self.keep_prob)

        self.test_logit, self.test_repre = \
            bag_attention(
                self.x_test, self.scope_ph, 
                self.ins_labels_ph, self.rel_tot, 
                False, keep_prob=1.0)
        self.test_probabs = tf.nn.softmax(self.test_logit)
        self.loss = softmax_cross_entropy(self.train_logit, self.labels_ph, self.rel_tot)

        self.optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)
        self.trainer = self.optimizer.minimize(self.loss)

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.saver = tf.train.Saver()


    def train_batch(self, 
        words, pos1, pos2, inst_rels, masks, lengths, rels, scope, keep_prob = 0.5):
        _, loss, x= self.sess.run([self.trainer, self.loss, self.train_logit],
            feed_dict = {self.words_ph : words,
        self.pos1_ph : pos1,
        self.pos2_ph : pos2,
        self.labels_ph : rels,
        self.ins_labels_ph : inst_rels,
        self.lengths_ph : lengths,
        self.scope_ph : scope,
        self.masks_ph : masks,
        self.keep_prob : keep_prob})
        return loss, x

    def test_batch(self, 
        words, pos1, pos2, inst_rels, masks, lengths, rels, scope):
        logits = self.sess.run(self.test_logit,
        feed_dict = {self.words_ph : words,
        self.pos1_ph : pos1,
        self.pos2_ph : pos2,
        self.labels_ph : rels,
        self.ins_labels_ph : inst_rels,
        self.lengths_ph : lengths,
        self.scope_ph : scope,
        self.masks_ph : masks,
        self.keep_prob : keep_prob})
        return logits

    def msaver(self, path):
        self.saver.save(self.sess, path)

