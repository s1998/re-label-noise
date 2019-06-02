import json
import numpy as np
import os

def get_preprocessed_dir(data_dir):
  return os.path.join(data_dir, "preprocessed")

def preprocess_word_vec(data_dir):
  file_name = os.path.join(data_dir, "word_vec.json")
  preprocessed_data_dir = get_preprocessed_dir(data_dir)
  wvecs_save_file = os.path.join(preprocessed_data_dir, "wordVecs.npy")
  wordId_save_file = os.path.join(preprocessed_data_dir, "word2Id.json")
  if os.path.exists(wvecs_save_file) and os.path.exists(wordId_save_file):
    print("Found preprocessed word vectors")
    return

  print("Preprocessing word to vector file")
  word2id = {}
  wordVecs = []
  with open(file_name, "r") as f:
    wvecs = json.load(f)
  for l in wvecs:
      word2id[l['word']] = len(wordVecs)
      wordVecs.append(l['vec'])
  count_of_words = len(wordVecs)
  word2id["<ukn>"] = count_of_words
  word2id["<start>"] = count_of_words + 1
  word2id["<end>"] = count_of_words + 2
  word2id["<blk>"] = count_of_words + 2
  wordVecs = np.array(wordVecs)
  np.save(wvecs_save_file, wordVecs)
  with open(wordId_save_file, 'w') as outfile:
    json.dump(word2id, outfile)

def load_word_vec(data_dir):
  preprocessed_data_dir = get_preprocessed_dir(data_dir)
  wvecs_save_file = os.path.join(preprocessed_data_dir, "wordVecs.npy")
  wordId_save_file = os.path.join(preprocessed_data_dir, "word2Id.json")
  if not os.path.exists(wvecs_save_file) or not os.path.exists(wordId_save_file):
    preprocess_word_vec(data_dir)
  wordVecs = np.load(wvecs_save_file)
  with open(wordId_save_file, 'r') as infile:
    word2Id = json.load(infile)
  return word2Id, wordVecs

def preprocessor_batch(data_dir, max_len = 120):
  files = ["train.json", "dev.json", "test.json"]
  preprocessed_data_dir = get_preprocessed_dir(data_dir)
  word2id, _ = load_word_vec(data_dir)
  with open(os.path.join(data_dir, "relToId.json"), "r") as f:
    relToId = json.load(f)

  def process_sent(sent):
    words = []
    sent["sentence"] = "<start> " + sent["sentence"] + " <end>"
    for word in sent['sentence'].split():
      if word.lower() in word2id:
        words.append(word2id[word.lower()])
      else:
        words.append(word2id["<ukn>"])

    words = words[:min(max_len, len(words))]
    sentence_len = len(words)
    words = words + [word2id["<blk>"]] * (max_len - len(words))

    try:
        ent1_p = min(int(sent['head']['start']) + 1, max_len - 1)
        ent2_p = min(int(sent['tail']['start']) + 1, max_len - 1)
    except Exception as ex:
        p1 = sent['sentence'].find(sent['head']['word'])
        p2 = sent['sentence'].find(sent['tail']['word'])
        if p1 != -1:
            ent1_p = len(sent['sentence'][:p1].split())
        else:
            ent1_p = 0
        if p2 != -1:
            ent2_p = len(sent['sentence'][:p2].split())
        else:
            ent2_p = 0

        
    pos1 = [i - ent1_p + max_len for i in range(max_len)]
    pos2 = [i - ent2_p + max_len for i in range(max_len)]
    mask = [0] * max_len
    pos_mini = min(ent1_p, ent2_p)
    pos_maxi = max(ent1_p, ent2_p)

    for i in range(max_len):
      if i < pos_mini:
        mask[i] = 1
      elif i < pos_maxi:
        mask[i] = 2
      elif i < sentence_len:
        mask[i] = 3
    try:
        rel = relToId[sent['relation']]
    except Exception as ex:
        rel = 0

    return words, pos1, pos2, mask, sentence_len, rel

  for file in files:
    print("Preprocessing file : ", file)
    fname = file[:-5]
    with open(os.path.join(data_dir, file), "r") as f:
      data = json.load(f)
    
    entPairRel = {}
    for sent in data:
      e1 = sent['head']['word']
      e2 = sent['tail']['word']
      try:
          rel = relToId[sent['relation']]
      except Exception as ex:
          rel = 0
      k = e1 + "#" + e2 + "#" + str(rel)
      if k in entPairRel:
        entPairRel[k].append((process_sent(sent)))
      else:
        entPairRel[k] = [(process_sent(sent))]

    all_words = []
    all_pos1 = []
    all_pos2 = []
    all_masks = []
    all_lengths = []
    all_inst_rels = []
    pair_bag_loc = {}

    start = 0
    for k in entPairRel:
      bag = entPairRel[k]
      no_sents = len(bag)
      end = start + no_sents
      pair_bag_loc[k] = (start, end)

      for instance in bag:
        words, pos1, pos2, mask, sentence_len, rel = instance
        all_words.append(words)
        all_pos1.append(pos1)
        all_pos2.append(pos2)
        all_masks.append(mask)
        all_lengths.append(sentence_len)
        all_inst_rels.append(rel)

      start = end
    
    all_words = np.array(all_words)
    print(all_words.shape)
    all_pos1 = np.array(all_pos1)
    print(all_pos1.shape)
    all_pos2 = np.array(all_pos2)
    print(all_pos2.shape)
    all_masks = np.array(all_masks)
    print(all_masks.shape)
    all_lengths = np.array(all_lengths)
    print(all_lengths.shape)
    all_inst_rels = np.array(all_inst_rels)
    print(all_inst_rels.shape)
    np.save(os.path.join(preprocessed_data_dir, str(fname) + "_words.npy"), all_words)
    np.save(os.path.join(preprocessed_data_dir, str(fname) + "_pos1.npy"), all_pos1)
    np.save(os.path.join(preprocessed_data_dir, str(fname) + "_pos2.npy"), all_pos2)
    np.save(os.path.join(preprocessed_data_dir, str(fname) + "_masks.npy"), all_masks)
    np.save(os.path.join(preprocessed_data_dir, str(fname) + "_lengths.npy"), all_lengths)
    np.save(os.path.join(preprocessed_data_dir, str(fname) + "_inst_rels.npy"), all_inst_rels)
    with open(os.path.join(preprocessed_data_dir, str(fname) + "_bag_locs.json"), "w") as f:
      json.dump(pair_bag_loc, f)


def load_data(preprocessed_data_dir, max_classes = 510, name = "train"):
  all_words = np.load(os.path.join(preprocessed_data_dir, name + "_words.npy"))
  all_pos1 = np.load(os.path.join(preprocessed_data_dir, name + "_pos1.npy"))
  all_pos2 = np.load(os.path.join(preprocessed_data_dir, name + "_pos2.npy"))
  all_masks = np.load(os.path.join(preprocessed_data_dir, name + "_masks.npy"))
  all_lengths = np.load(os.path.join(preprocessed_data_dir, name + "_lengths.npy"))
  all_inst_rels = np.load(os.path.join(preprocessed_data_dir, name + "_inst_rels.npy"))
  for i in range(all_inst_rels.shape[0]):
    if all_inst_rels[i] > max_classes:
      all_inst_rels[i] = 0
  with open(os.path.join(preprocessed_data_dir, name + "_bag_locs.json"), "r") as f:
    pair_bag_loc = json.load(f)
  remove_keys = []
  
  for k in list(pair_bag_loc.keys()):
    if int(k.split("#")[2]) > max_classes:
      pair_bag_loc["#".join(k.split("#")[:2] + ["0"])] = pair_bag_loc[k]
      remove_keys.append(k)
  
  for k in remove_keys:
    pair_bag_loc.pop(k, None)
  
  return all_words, all_pos1, all_pos2, all_masks, all_lengths, all_inst_rels, pair_bag_loc

def batch_maker(data, batch_keys):
  all_words, all_pos1, all_pos2, all_masks, all_lengths, \
    all_inst_rels, pair_bag_loc = data
  words = []
  pos1 = []
  pos2 = []
  inst_rels = []
  masks = []
  lengths = []
  rels = []
  scope = []

  curr_start = 0  
  curr_end = 0

  for k in batch_keys:
    start, end = pair_bag_loc[k]
    curr_end = curr_start + end -start
    words.append(all_words[start:end])
    pos1.append(all_pos1[start:end])
    pos2.append(all_pos2[start:end])
    masks.append(all_masks[start:end])
    inst_rels.append(all_inst_rels[start:end])
    lengths.append(all_lengths[start:end])
    rels.append(all_inst_rels[start])
    scope.append([curr_start,curr_end])
    curr_start = curr_end

  words = np.concatenate(words)
  pos1 = np.concatenate(pos1)
  pos2 = np.concatenate(pos2)
  inst_rels = np.concatenate(inst_rels)
  masks = np.concatenate(masks)
  lengths = np.concatenate(lengths)
  rels = np.array(rels)
  scope = np.array(scope)

  return words, pos1, pos2, inst_rels, masks, lengths, rels, scope

def get_dev_pairs_dict(dev_pairs):
  dev_pairs_dict = {}
  for k in dev_pairs:
    ent_pair = "#".join(k.split("#")[:2])
    rel = int(k.split("#")[2])
    if ent_pair in dev_pairs_dict:
      dev_pairs_dict[ent_pair].append(rel)
    else:
      dev_pairs_dict[ent_pair] = [rel]
  return dev_pairs_dict
