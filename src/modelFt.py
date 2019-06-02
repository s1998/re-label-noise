from src.model_utils import *
from src.utils import load_word_vec, batch_maker
import tensorflow as tf
import numpy as np

class ModelFt:
    def __init__(self,
        word_vec_mat,
        encoder = "pcnn", selector="att", no_of_classes = 503, 
        l2_lambda = 0.01, bs_type="none", bs_val=0.0, learning_rate = 0.001, load_logit = True): 
        print("Creating model with encoder and selector : ", encoder, selector)
        self.encoder = encoder
        self.selector = selector
        self.words_ph = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='word')
        self.pos1_ph = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos1')
        self.pos2_ph = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos2')
        self.labels_ph = tf.placeholder(dtype=tf.int64, shape=[batch_size], name='label')
        self.ins_labels_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='ins_label')
        self.lengths_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='length')
        self.scope_ph = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2], name='scope')
        self.masks_ph = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="mask")
        self.learning_rate = learning_rate
        self.learning_rate_ph = tf.placeholder(tf.float32)
        self.rel_tot = no_of_classes
        self.keep_prob = tf.placeholder(tf.float32)
        self.l2_lambda = l2_lambda

        self.x = word_position_embedding(
            self.words_ph, word_vec_mat, self.pos1_ph, self.pos2_ph)

        if encoder == "pcnn":
            self.x_train = pcnn(self.x, self.masks_ph, keep_prob=0.5)
            self.x_test = pcnn(self.x, self.masks_ph, keep_prob=1.0)
        elif encoder == "pcnn2":
            self.x_train = pcnn2(self.x, self.masks_ph, keep_prob=0.5)
            self.x_test = pcnn2(self.x, self.masks_ph, keep_prob=1.0)
        elif encoder == "pcnn2n":
            self.x_train = pcnn2n(self.x, self.masks_ph, keep_prob=0.5)
            self.x_test = pcnn2n(self.x, self.masks_ph, keep_prob=1.0)
        elif encoder == "brnn":
            self.x_train = birnn(self.x, self.lengths_ph, keep_prob=0.5)
            self.x_test = birnn(self.x, self.lengths_ph, keep_prob=1.0)
        elif encoder == "bgwa":
            self.x_train = bgwa(self.x, self.lengths_ph, keep_prob=0.5)
            self.x_test = bgwa(self.x, self.lengths_ph, keep_prob=1.0)
        else:
            raise Exception("Encoder not implemented : --{}--".format(encoder))

        print(self.x_train.shape)

        if selector == "att":
          self.train_logit, self.train_repre, self.att_scores = bag_attention(
              self.x_train, self.scope_ph, 
              self.ins_labels_ph, self.rel_tot, True, keep_prob=0.5)

          self.test_logit, self.test_repre, self.att_scores = \
              bag_attention(
                  self.x_test, self.scope_ph, 
                  self.ins_labels_ph, self.rel_tot, 
                  False, keep_prob=1.0)
          self.test_probabs = self.test_logit
        elif selector == "cross_sent_max":
          self.train_logit, self.train_repre, self.att_scores = bag_cross_max(
              self.x_train, self.scope_ph, 
              self.rel_tot, True, keep_prob=0.5)

          self.test_logit, self.test_repre, self.att_scores = \
              bag_cross_max(
                  self.x_test, self.scope_ph, 
                  self.rel_tot, 
                  False, keep_prob=1.0)
          self.test_probabs = tf.nn.softmax(self.test_logit)
        else:
          raise Excetion("Selector not defined")

        if bs_type is "none":
          self.loss = softmax_cross_entropy(self.train_logit, 
            tf.one_hot(self.labels_ph, self.rel_tot), self.rel_tot)
          print("Created model with no bootstrapping, bs val : {}".format(bs_val))
        else:
          raise Exception('Bootstrapping method not present')

        self.l2_loss = tf.constant(0.0)
        
        intermediate_vars = [self.x_train, self.x_test, self.train_logit, self.train_repre, \
          self.test_logit, self.test_repre, self.loss, self.l2_loss]

        self.loss += self.l2_loss

        # self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate_ph)
        # self.trainer = self.optimizer.minimize(self.loss)
        
        # self.gvs = self.optimizer.compute_gradients(self.loss)
        # self.ft_grads = []
        train_vars_last = []
        train_vars_others = []
        for kv in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
          if "logit" in kv.name:
            train_vars_last.append(kv)
          else:
            train_vars_others.append(kv)
        print("Train vars last : ", train_vars_last)
        print("Train vars others : ", train_vars_others)
        
        self.var_list1 = train_vars_last
        self.var_list2 = train_vars_others
        self.opt1 = tf.train.AdamOptimizer(self.learning_rate_ph)
        self.opt2 = tf.train.AdamOptimizer(self.learning_rate_ph / 2.5)
        self.grads = tf.gradients(self.loss, self.var_list1 + self.var_list2)
        self.grads1 = self.grads[:len(self.var_list1)]
        self.grads2 = self.grads[len(self.var_list1):]
        self.train_op1 = self.opt1.apply_gradients(zip(self.grads1, self.var_list1))
        self.train_op2 = self.opt2.apply_gradients(zip(self.grads2, self.var_list2))
        self.train_op = tf.group(self.train_op1, self.train_op2)
      
        self.adam_vars = [kv for kv in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
          if "adam" in kv.name.lower()]
        self.adam_reset = [var.initializer for var in self.adam_vars]

        self.sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./summ/train',
                                            self.sess.graph)
        self.test_writer = tf.summary.FileWriter('./summ/test')
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
        if load_logit is True:
          self.loader = tf.train.Saver([kv for kv in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
           if ("adam" not in kv.name.lower() and "beta" not in kv.name)])
        else:
          self.loader = tf.train.Saver([kv for kv in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
           if ("logit" not in kv.name and "adam" not in kv.name.lower() and "beta" not in kv.name)])
        
        self.saver = tf.train.Saver()

    def train_batch(self, 
        words, pos1, pos2, inst_rels, masks, lengths, rels, scope, learning_rate = None):
        if learning_rate is None:
          learning_rate = self.learning_rate
        summary, _, loss, x= self.sess.run([self.merged, self.train_op, self.loss, self.train_logit],
            feed_dict = {self.words_ph : words,
              self.pos1_ph : pos1,
              self.pos2_ph : pos2,
              self.labels_ph : rels,
              self.ins_labels_ph : inst_rels,
              self.lengths_ph : lengths,
              self.scope_ph : scope,
              self.masks_ph : masks,
              self.learning_rate_ph : learning_rate,
              self.keep_prob : 0.5})
        self.train_writer.add_summary(summary)
        return loss, x

    def test_batch(self, 
        words, pos1, pos2, inst_rels, masks, lengths, rels, scope, learning_rate = None):
        if learning_rate is None:
          learning_rate = self.learning_rate
        summary, probabs, att_scores = self.sess.run([self.merged, self.test_probabs, self.att_scores],
            feed_dict = {self.words_ph : words,
              self.pos1_ph : pos1,
              self.pos2_ph : pos2,
              self.labels_ph : rels,
              self.ins_labels_ph : inst_rels,
              self.lengths_ph : lengths,
              self.scope_ph : scope,
              self.masks_ph : masks,
              self.learning_rate_ph : learning_rate,
              self.keep_prob : 1.0})
        self.test_writer.add_summary(summary)
        return probabs, att_scores

    def msaver(self, path):
        print("Saved model at path : {}".format(path))
        self.saver.save(self.sess, path)

    def mloader(self, path):
        print("Loded model from path : {}".format(path))
        self.loader.restore(self.sess, path)

    def reset_optimizer(self):
      print("Resetting adam optimizer variables.")
      self.sess.run(self.adam_reset)

    def print_lr_los(self, pair_batches, train_data, no, start = 1e-9):
      print("Saving temp model")
      self.saver.save(self.sess, "/tmp/abcd")
      lrs = []
      losses = []
      for i in range(no):
        lr = start * (1.2 ** (i))
        loss_t = []
        for j in range(5):
          batch_keys = pair_batches[(i * 5 + j)]
          words, pos1, pos2, inst_rels, masks, lengths, \
            rels, scope = batch_maker(train_data, batch_keys)
          loss_, _ = self.train_batch(
            words, pos1, pos2, inst_rels, masks, lengths, rels, scope)
          loss_t.append(loss_)

        lrs.append(lr)
        losses.append(sum(loss_t) / 5.0)
      print("Restoring temp model")
      self.saver.restore(self.sess, "/tmp/abcd")
      return lrs, losses
    
    def print_extra_layer(self):
      return self.sess.run([self.extra_layer])

    def train_batch_ft(self, 
        words, pos1, pos2, inst_rels, masks, lengths, rels, scope, learning_rate = None):
        if learning_rate is None:
          learning_rate = self.learning_rate
        summary, _, loss, x= self.sess.run([self.merged, self.train_op, self.loss, self.train_logit],
            feed_dict = {self.words_ph : words,
              self.pos1_ph : pos1,
              self.pos2_ph : pos2,
              self.labels_ph : rels,
              self.ins_labels_ph : inst_rels,
              self.lengths_ph : lengths,
              self.scope_ph : scope,
              self.masks_ph : masks,
              self.learning_rate_ph : learning_rate,
              self.keep_prob : 0.5})
        self.train_writer.add_summary(summary)
        return loss, x

