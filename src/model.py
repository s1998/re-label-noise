from src.model_utils import *
from src.utils import load_word_vec, batch_maker
import tensorflow as tf
import numpy as np

class Model:
    def __init__(self,
        word_vec_mat,
        encoder = "pcnn", selector="att", no_of_classes = 503, 
        l2_lambda = 0.01, bs_type="none", bs_val=0.0, learning_rate = 0.001, tl = False): 
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
        if encoder in all_encoders:
        	self.x_train = all_encoders[encoder](self.x, self.lengths_ph, self.masks_ph, keep_prob=0.5)
        	self.x_test = all_encoders[encoder](self.x, self.lengths_ph, self.masks_ph, keep_prob=1.0)
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

        if bs_type in all_losses:
        	self.loss = all_losses[bs_type](self.train_logit, self.labels_ph, self.rel_tot, bs_val, l2_lambda)
        else:
        	raise Exception('Bootstrapping method not present')

        self.l2_loss = tf.constant(0.0)
        self.loss += self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate_ph)
        self.trainer = self.optimizer.minimize(self.loss)

        self.logit_vars = [kv for kv in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
          if "logit" in kv.name.lower()]
        
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
        if bs_type == "extra":
          self.loader = tf.train.Saver([kv for kv in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if "extra" not in kv.name])
        elif tl is True:
          self.loader = tf.train.Saver([kv for kv in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
           if ("logit" not in kv.name and "adam" not in kv.name.lower() and "beta" not in kv.name)])
        else:
          self.loader = tf.train.Saver()

        self.saver = tf.train.Saver()

    def train_batch(self, 
        words, pos1, pos2, inst_rels, masks, lengths, rels, scope, learning_rate = None):
        if learning_rate is None:
          learning_rate = self.learning_rate
        summary, _, loss, x= self.sess.run([self.merged, self.trainer, self.loss, self.train_logit],
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

    def train_batch_l(self, 
        words, pos1, pos2, inst_rels, masks, lengths, rels, scope, learning_rate = None):
        if learning_rate is None:
          learning_rate = self.learning_rate
        summary, _, loss, x= self.sess.run([self.merged, self.trainer_logit, self.loss, self.train_logit],
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

    def print_lr_los(self, pair_batches, train_data, no):
      print("Saving temp model")
      self.saver.save(self.sess, "/tmp/abcd")
      lrs = []
      losses = []
      for i in range(no):
        lr = 1e-5 * (1.1 ** i)
        batch_keys = pair_batches[i]
        words, pos1, pos2, inst_rels, masks, lengths, \
          rels, scope = batch_maker(train_data, batch_keys)
        loss_, _ = self.train_batch(
          words, pos1, pos2, inst_rels, masks, lengths, rels, scope)
        lrs.append(lr)
        losses.append(loss_)
      print("Restoring temp model")
      self.saver.restore(self.sess, "/tmp/abcd")
      return lrs, losses
    
    def print_extra_layer(self):
      return self.sess.run([self.extra_layer])

    def get_repre(self, 
        words, pos1, pos2, inst_rels, masks, lengths, rels, scope, learning_rate = None):
        if learning_rate is None:
          learning_rate = self.learning_rate
        repre, probs = self.sess.run([self.test_repre, self.test_probabs],
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
        return repre, probs

