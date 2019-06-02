# Relation extraction modelling label noise
Code for modelling label noie in distantly supervised relation extraction.

# Usage help

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
# Results

Results on NYT dataset

## Results for cross senence maxpooling on NYT dataset

```
Encoder  | Selector       |  AUC
---------------------------------
pcnn     | att            | 0.338
rnn      | att            | 0.333
brnn     | att            | 0.344
pcnn     | cross-sent-max | 0.369
rnn      | cross-sent-max | 0.385
brnn     | cross-sent-max | 0.383
```

## Results for modelling label noise on NYT dataset

```
Mechanism                  | AUC
----------------------------------
PCNN + ATT                 | 0.338
PCNN + ATT + extra_layer   | 0.348
```

We found that value of AUC score decreased for bootsrapping methods both hard and soft.


# References 
1. [THUNLP's relation extraction](https://github.com/thunlp/OpenNRE)
2. [Adversarial methods for relation extraction](https://github.com/jxwuyi/AtNRE)
3. [Soft label methods fro relation extraction](https://github.com/tyliupku/soft-label-RE)



