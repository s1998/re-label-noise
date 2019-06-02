from random import shuffle
from src.model import *
from src.utils import *

import argparse
import itertools
import os
import random
import sklearn.metrics
import sys

parser = argparse.ArgumentParser(description='Running code for relation extraction.')
parser.add_argument('--encoder', default='pcnn', help='select the encoders from \
	pcnn ,pbrnn ,pcnn2 (stacked pcnn) ,rnn ,brnn ,crnn ,crnn2 ,bgwa')
parser.add_argument('--selector', default='att', help='select the bag selector from \
	att, cross_sent_max')
parser.add_argument('--loss_type', default='none', help='select the loss type from \
	none, extra (layer for noise modelling), hard (bootstrapping), soft (bootstrapping)')
parser.add_argument('--bs_val', type=float, default=0.0, help='select the bootstrapping value\
	(only valid if loss is of type hard/soft)')
parser.add_argument('--dataset', default='nyt', help='select the dataset from nyt/wiki')
parser.add_argument('--chkpt_pt', default='', help='path to saved model \
	(empty if no checkpoint to load from)')
parser.add_argument('--l2_val', type=float, default=0.0, help='l2 lambda value')
args = parser.parse_args()

best_auc = 0.0
best_test_auc = 0.0

def trainer(bs_type = "none", 
        bs_val = 0, 
        l2_lambd = 0, 
        check_pt = "", 
        undersample_na = False, 
        encoder = "pcnn",
        selector = "att",
        dataset_name = "nyt",
        learning_rate = 0.001, 
        tl = False):
  if dataset_name == "nyt":
    max_classes = 53
  elif dataset_name == "nytf":
    max_classes = 53
  elif dataset_name == "nyt2":
    max_classes = 53
  elif dataset_name == "wikif":
    max_classes = 108
  elif dataset_name == "wiki":
    max_classes = 108 
  else:
    raise Exception("Dataset not found error.")

  data_dir = os.path.join("data", dataset_name)
  fname_prefix = encoder + "_" + selector + "_" + dataset_name + "_" + bs_type + "_" + \
    str(max_classes) + "_n_" + str(l2_lambd) + "_"
  if undersample_na:
    fname_prefix += "undersample_"
  if tl:
    fname_prefix += "tl_"
  global best_auc
  global best_test_auc
  best_auc = 0.1
        
  preprocessed_data_dir = get_preprocessed_dir(data_dir)
  if not os.path.exists(preprocessed_data_dir):
    os.mkdir(preprocessed_data_dir)
    preprocessor_batch(data_dir)


  tf.reset_default_graph()
  _, word_vec_mat = load_word_vec(os.path.join("data", dataset_name))
  model = Model(
    word_vec_mat, 
    encoder = encoder, 
    selector = selector, 
    l2_lambda = l2_lambd, 
    bs_type = bs_type, 
    bs_val = bs_val, 
    no_of_classes = max_classes,
    learning_rate = learning_rate, 
    tl = tl)
  if check_pt is not "":
    model.mloader(os.path.join("saved", "models", check_pt))
    model.reset_optimizer()
  print("Setting max class size to : ", max_classes)
  train_data = load_data(preprocessed_data_dir, max_classes)
  test_data = load_data(preprocessed_data_dir, max_classes, "test")
  if dataset_name == "nyt2":
    dev_data = load_data(preprocessed_data_dir, max_classes, "test")
  else:
    dev_data = load_data(preprocessed_data_dir, max_classes, "dev")

  n_epochs = 2
  n_epoch_onl_logits = 3
  pair_bag_loc = train_data[-1]
  pairs = list(pair_bag_loc.keys())
  
  dev_pair_bag_loc = dev_data[-1]
  dev_pairs = list(dev_pair_bag_loc.keys())
  dev_pairs_dict = get_dev_pairs_dict(dev_pairs)
  n_dev_batches = len(dev_pairs) // batch_size
  
  test_pair_bag_loc = test_data[-1]
  test_pairs = list(test_pair_bag_loc.keys())
  test_pairs_dict = get_dev_pairs_dict(test_pairs)
  n_test_batches = len(test_pairs) // batch_size

  def na_nonNA(pairs):
    not_NA_rels = 0
    naPairs = []
    nonNaPairs = []
    for k in pairs:
      if k.split("#")[2] != "0":
        nonNaPairs.append(k)
      else:
        naPairs.append(k)
    return naPairs, nonNaPairs

  trainNa, trainNonNa = na_nonNA(pairs)
  devNa, devNonNa = na_nonNA(dev_pairs)
  testNa, testNonNa = na_nonNA(test_pairs)
  not_NA_rels = len(devNonNa)
  print("Pairs in train dataset NA and non NA : ", len(trainNa), len(trainNonNa))
  print("Pairs in dev dataset NA and non NA : ", len(devNa), len(devNonNa))
  print("Pairs in test dataset NA and non NA : ", len(testNa), len(testNonNa))

  if undersample_na:
    n_batches = 2 *  len(trainNonNa) // batch_size
    print("No of train batches : ", n_batches)
  else:
    n_batches = len(pairs) // batch_size
    print("No of train batches : ", n_batches)

  def lr_vs_loss():
    random.shuffle(pairs)
    split_pairs = [pairs[i * batch_size : (i + 1) * batch_size] for i in range(100)]
    print(model.print_lr_los(split_pairs, train_data, 100))


  def train_epoch(epoch_no):
    losses = []
    aucs = []
    # lr_vs_loss()
    # learning_rate = input()
    # learning_rate = float(learning_rate) 
    if epoch_no > 3:
      learning_rate = 0.0001
    else:
      learning_rate = 0.001
    print("Setting learning rate for the epoch no {} to : {}".format(epoch_no, learning_rate))
    if undersample_na or (tl and epoch_no < n_epoch_onl_logits):
      random.shuffle(trainNa)
      curr_pairs = trainNonNa + trainNa[:1 * len(trainNonNa)]
      random.shuffle(curr_pairs)
    else:
      curr_pairs = pairs

    n_batches = len(curr_pairs) // batch_size
    print("No of batches : {} \n ".format(n_batches))
    for i in range(n_batches):
      if i % 100 == 0:
        sys.stdout.write("\033[K")
        print("Running for batch : {}/{}".format(i + 1, n_batches))
        
      batch_keys = curr_pairs[i * batch_size : (i + 1) * batch_size]
      words, pos1, pos2, inst_rels, masks, lengths, \
        rels, scope = batch_maker(train_data, batch_keys)
      
      if epoch_no < n_epoch_onl_logits and i < 4000 and tl:
        if i % 100 == 0:
          print("Training only the logit layer.")
        loss_, x= model.train_batch_l(
          words, pos1, pos2, inst_rels, masks, lengths,
          rels, scope, learning_rate)
      else:
        loss_, x= model.train_batch(
          words, pos1, pos2, inst_rels, masks, lengths,
          rels, scope, learning_rate)

      if loss_ > 10:
        temp = []
        for i in range(x.shape[0]):
          for j in range(x.shape[1]):
            temp.append((x[i][j], i, j))
        print(sorted(temp, reverse = True)[:100])
        print(batch_keys[i])

      if i%500 == 250:
        losses.append(loss_)
        auc1 = test_model(n_test_batches, test_pairs, test_pairs_dict, test_data, True)
        print("Test AUC : ", auc1)
        if dataset_name != "nyt2":
          auc2 = test_model(n_dev_batches, dev_pairs, dev_pairs_dict, dev_data)
          print("Auc : ", auc2)
        else:
          auc2 = auc1
        print("Dev AUC : ", auc2)
        aucs.append((auc1, auc2))
        #print(x)
        #print([k for k in x[0].tolist()])

    print("Testing after epoch : ")
    auc1 = test_model(n_test_batches, test_pairs, test_pairs_dict, test_data)
    print("Test Auc : ", auc1)
    if dataset_name != "nyt2":
      auc2 = test_model(n_dev_batches, dev_pairs, dev_pairs_dict, dev_data)
      print("Auc : ", auc2)
    else:
      auc2 = auc1
    aucs.append((auc1, auc2))
    return losses, aucs

  def test_model(batches, pairs, pairs_dict, processed_data, print_vals = False):
    test_res = []
    for i in range(batches):
      batch_keys = pairs[i * batch_size : (i + 1) * batch_size]
      words, pos1, pos2, inst_rels, masks, lengths, \
        rels, scope = batch_maker(processed_data, batch_keys)
      
      output, _ = model.test_batch(words, pos1, pos2, inst_rels, 
        masks, lengths, rels, scope)

      for i, k in enumerate(batch_keys):
        entPair = "#".join(k.split("#")[:2])
        entPairRels = pairs_dict[entPair]
        for j in range(1, max_classes):
          correct = 0
          if j in entPairRels:
            correct = 1
          test_res.append({"entPair" : entPair, "actual" : entPairRels, "predicted" : j,
            "score" : output[i][j], "correct" : correct})
    prec = []
    recall = []
    correct = 0
    sorted_test_result = sorted(test_res, key=lambda x: x['score'], reverse = True)

    import time
    start = time.time()
    if print_vals:
      for item in sorted_test_result[:100]:
        print(item["entPair"], item["actual"], item["predicted"], end=' -!-')

    
    for i, item in enumerate(sorted_test_result):
      if item["correct"]:
        correct += 1  
      prec.append(float(correct) / (i + 1))
      recall.append(float(correct) / not_NA_rels)
    auc = sklearn.metrics.auc(x = recall, y = prec)
    end = time.time()
    #print("Time to calc : ", end - start)
    global best_auc
    if auc > best_auc:
      fname = fname_prefix + str(best_auc)[:6]
      if os.path.exists(fname):
        os.remove(fanme) 
      best_auc = auc
      fname = fname_prefix + str(auc)[:6] 
      print("Saving model {}".format(fname))
      model.msaver(os.path.join("saved", "models", fname))
    return auc


  import pickle
  all_losses = {}
  all_aucs = {}
  best_auc = 0.1
  for n in range(n_epochs):
    shuffle(pairs)
    shuffle(dev_pairs)
    print("Running epoch no : ", n)
    all_losses[n], all_aucs[n] = train_epoch(n)

  print(all_losses, all_aucs)
  best_dev_auc = 0
  best_test_auc = 0
  for k in range(n_epochs):
    print(all_aucs[k])
    for auc_test, auc_dev in all_aucs[k]:
      if auc_dev > best_dev_auc:
        best_dev_auc = auc_dev
        best_test_auc = auc_test

  
  # with open(os.path.join(data_dir, "result", "loss.pkl"), "wb") as f:
  #   pickle.dump((all_losses, all_aucs), f)
  # return {'loss': -1.0 * best_auc, 'status': STATUS_OK }
  return best_dev_auc, best_test_auc

# Training the modle and obtaining the auc score
bs_type = args.bs_type
enc = args.encoder
sel = args.selector
dataset_name = args.dataset_name
bs_val = args.bs_val
chkpt_pt = args.chkpt_pt
l2_lambd = args.l2_val

auc = trainer(bs_type = bs_type, bs_val = bs_val, check_pt = args.chkpt_pt, encoder = enc, selector = sel, l2_lambd = l2_lambd
	dataset_name = dataset_name):
print("Best auc value : {}".format(auc))

