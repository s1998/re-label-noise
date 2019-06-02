from src.model_utils import *
from src.utils import load_word_vec
import tensorflow as tf
import numpy as np

class ModelForLayerWiseTl:
    def __init__(self,
        word_vec_mat,
        encoder = "pcnn", selector="att", no_of_classes = 503, 
        l2_lambda = 0.01, bs_type="none", bs_val=0.0): 
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
        self.rel_tot = no_of_classes
        self.keep_prob = tf.placeholder(tf.float32)
        self.l2_lambda = l2_lambda

        self.x = word_position_embedding(
            self.words_ph, word_vec_mat, self.pos1_ph, self.pos2_ph)

        if encoder == "pcnn2":
            self.x_train = pcnn2tl(self.x, self.masks_ph, keep_prob=0.5)
            self.x_test = pcnn2tl(self.x, self.masks_ph, keep_prob=1.0)
        else:
            raise Exception("Encoder not implemented : --{}--".format(encoder))

        if selector == "cross_sent_max":
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

        self.loss = softmax_cross_entropy(self.train_logit, 
            tf.one_hot(self.labels_ph, self.rel_tot), self.rel_tot)

        self.l2_loss = tf.constant(0.0)
        for kv in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
          k = kv.name
          print(k)
 
        intermediate_vars = [self.x_train, self.x_test, self.train_logit, self.train_repre, \
          self.test_logit, self.test_repre, self.loss, self.l2_loss]

        for var in intermediate_vars:
          #variable_summaries(var)
          pass

        self.loss += self.l2_loss
        
        self.logit_vars = [kv for kv in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
          if "logit" in kv.name.lower()]
        
        self.optimizer_logit = tf.train.AdamOptimizer(learning_rate = 1e-3)
        self.trainer_logit = self.optimizer_logit.minimize(self.loss)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)
        self.trainer = self.optimizer.minimize(self.loss)
        
        self.adam_vars = [kv for kv in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
          if "adam" in kv.name.lower()]
        # print(self.adam_vars)
        self.adam_reset = [var.initializer for var in self.adam_vars]

        self.sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./summ/train',
                                            self.sess.graph)
        self.test_writer = tf.summary.FileWriter('./summ/test')
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.saver = tf.train.Saver()


    def train_batch(self, 
        words, pos1, pos2, inst_rels, masks, lengths, rels, scope):
        summary, _, loss, x= self.sess.run([self.merged, self.trainer, self.loss, self.train_logit],
            feed_dict = {self.words_ph : words,
              self.pos1_ph : pos1,
              self.pos2_ph : pos2,
              self.labels_ph : rels,
              self.ins_labels_ph : inst_rels,
              self.lengths_ph : lengths,
              self.scope_ph : scope,
              self.masks_ph : masks,
              self.keep_prob : 0.5})
        self.train_writer.add_summary(summary)
        return loss, x

    def test_batch(self, 
        words, pos1, pos2, inst_rels, masks, lengths, rels, scope):
        summary, probabs, att_scores = self.sess.run([self.merged, self.test_probabs, self.att_scores],
            feed_dict = {self.words_ph : words,
              self.pos1_ph : pos1,
              self.pos2_ph : pos2,
              self.labels_ph : rels,
              self.ins_labels_ph : inst_rels,
              self.lengths_ph : lengths,
              self.scope_ph : scope,
              self.masks_ph : masks,
              self.keep_prob : 1.0})
        self.test_writer.add_summary(summary)
        return probabs, att_scores

    def msaver(self, path):
        print("path : {}".format(path))
        self.saver.save(self.sess, path)

    def mloader(self, path):
        print("path : {}".format(path))
        self.saver.restore(self.sess, path)

    def reset_optimizer(self):
      print("Resetting adam optimizer variables.")
      self.sess.run(self.adam_reset)

