# re-label-noise
Code for modelling label noie in distantly supervised relation extraction

# Sample usage

```
usage: main.py [-h] [--encoder ENCODER] [--selector SELECTOR]
               [--loss_type LOSS_TYPE] [--bs_val BS_VAL] [--dataset DATASET]
               [--chkpt_pt CHKPT_PT] [--l2_val L2_VAL]

Running code for relation extraction.

optional arguments:
  -h, --help            show this help message and exit
  --encoder ENCODER     select the encoders from pcnn ,pbrnn ,pcnn2 (stacked
                        pcnn) ,rnn ,brnn ,crnn ,crnn2 ,bgwa
  --selector SELECTOR   select the bag selector from att, cross_sent_max
  --loss_type LOSS_TYPE
                        select the loss type from none, extra (layer for noise
                        modelling), hard (bootstrapping), soft (bootstrapping)
  --bs_val BS_VAL       select the bootstrapping value (only valid if loss is
                        of type hard/soft)
  --dataset DATASET     select the dataset from nyt/wiki
  --chkpt_pt CHKPT_PT   path to saved model (empty if no checkpoint to load
                        from)
  --l2_val L2_VAL       l2 lambda value
```
