## General parameters

### task : *{'train', 'test'}*

Whether the current experiment is for training or testing.

### exp_dir : *str*

### data_dirs : *list of str*

A list of directories, where each directory represents a scenario and includes a training dataset named as `train.h5` and a testing dataset named as `test.h5`.

### scheme : *{'ad_cnn', 'ad_cnn_multitask', 'cnn', 'mfc_net', 'alloc', 'iclloc'}*

### iclloc_scheme : *{'generative', 'discriminative'}, optional*

The training scheme of ICLLoc. Ignored if *scheme* is not 'iclloc'.

### num_ant : *int*

Number of antennas. Default: 32.

### num_car : *int*

Number of subcarriers. Default: 32.

### max_len_nei : *int*

Number of total neighbors. Default: 64.

### select : *{'random', 'neighbor'}, optional*

For ALLoc/ICLLoc, whether to apply random sampling ('random') or neighborhood sampling ('neighbor') when generating the neighbors.

### radius : *float or None, optional*

For ALLoc/ICLLoc applying neighborhood sampling, this parameter determines how to select neighboring samples:

- If *radius* is a value, neighbors are randomly selected from training samples within a distance of *radius* (m) from the to-be-inferred sample.
- If *radius* is None, neighbors are the nearest training samples in the whole dataset to the to-be-inferred sample.

Default: None.

In our experiments, *radius* is set as follows:

- DICHASUS-ca0x: 1.
- Others: None (default value).

### inv_std : *float*

Multiplicative factor of CSI data before feeding into LNNs. Default: 10000.

In our experiments, *inv_std* is set as follows:

- DeepMIMO O1, O1B, MO1: 10000 (default value).
- DeepMIMO RO1, O2: 100000.
- DICHASUS-ca0x: 0.1.
- Sionna RT: 10000 (default value).

### num_train_data : *int*

Number of training data for each scenario. Default: 40000.

### num_test_data : *int*

Number of testing data for each scenario. Default: 20000.

In our experiments, ca02 serves as an additional testing trajectory for DICHASUS-ca0x, which has 10000 testing data.

## Training-specific parameters

### tensorboard_writer : *bool*

Default: True.

### num_epochs : *int*

Total number of training epochs. The training process ends if the number of training steps exceeds *num_max_steps* or the number of training epochs exceeds *num_epochs*. Default: 100000.

### num_max_steps : *int*

Total number of training steps. The training process ends if the number of training steps exceeds *num_max_steps* or the number of training epochs exceeds *num_epochs*. Default: 300000.

### init_lr : *float*

Initial learning rate. Default: 0.0001.

In our experiments, *init_lr* is set as follows:

- ALLoc, ICLLoc: 0.0001 (default value).
- MFCNet, CNN, AD_CNN (including multi-task learning): 0.001.
- Transfer learning: 0.0001 (default value).

### lr_decay_steps : *list of int*

At which training steps the learning rate is decayed by a multiplicative factor. Default: [150000,200000,250000].

### lr_decay_factor : *float*

Multiplicative factor to decay the learning rate. Default: 0.2.

### batch_size : *int*

Default: 500.

### mix_train_datasets : *bool*

For multi-scenario learning with *N* scenarios, this parameter determines how the training datasets from different scenarios are mixed during the training process.

- If *mix_train_datasets* is True, every *N* training steps exactly load one batch from each scenario, referred to as batch-wise mixing.
- If *mix_train_datasets* is False, all training batches from one scenario are loaded in consecutive training steps, referred to as dataset-wise mixing.

In both cases, training samples in a single batch come from the same scenario, i.e., sample-wise mixing is not supported.

Default: True.

### gen_seq_datasets : *bool, optional*

For ALLoc/ICLLoc, this parameter determines how the neighbors are generated during the training process.

- If *gen_seq_datasets* is True, the neighbors of each data sample are generated in advance and saved in a sequence dataset. In this sequence dataset, each sample is a sequence of length *max_len_nei* + 1, where the last item is the original data sample and the remaining *max_len_nei* items are its neighbors. Such sequence datasets might require huge storage, so make sure that the storage space is sufficient when setting this parameter as True.
- If *gen_seq_datasets* is False, the neighbors of each data sample are generated temporarily each time this sample is used. No extra dataset needs to be stored, but the training process will be slow.

Default: True.

Note that sequence datasets are genereted and stored when *task* is 'train' only. When *task* is 'test', neighbors are always generated temporarily, in order to cope with various testing conditions such as inaccurate initial locations and iterative search.

### len_nei_total : *int*

Number of total neighbors. Default: 64.

### len_nei_least : *int*

The least number of samples for neighborhood embedding during training. Default: 16.

### len_nei_test : *int*

Number of sampled neighbors during testing. Default: 64.

### log_step_inter : *int*

Default: 100.

### save_epoch_inter : *int*

Default: 100.

### test_epoch_inter : *int*

Default: 20.

### finetune : *bool*

This parameter is for transfer learning and multitask learning purposes.

- If *finetune* is True, an existing model specified by the *finetune_model.exp_dir* and *finetune_model.epoch* parameters will be loaded as the initial model. For multi-task learning, i.e., *scheme* is 'ad_cnn_multitask', only the backbone will be loaded. 
- If *finetune* is False, the model is randomly initialized as usual.

Default: False.

### finetune_model.exp_dir : *str, optional*

The directory where the initial model is loaded from. This parameter should be the *exp_dir* parameter of a previous experiment whose *task* parameter is 'train'. 

Ignored if *finetune* is False.

### finetune_model.epoch : *int or None, optional*

- If *finetune_model.epoch* is an integer *E*, the model after *E* training epochs will be loaded from checkpoint.
- If *finetune_model.epoch* is None, the final model will be loaded.

Ignored if *finetune* is False.

## Testing-specific parameters

### batch_size : *int*

Default: 1.

### len_nei_test : *int*

Number of sampled neighbors during testing. Default: 64.

### noise_dev : *float*

The intensity of noise disturbance, representing the deviation between the noisy and ideal channels. Default: 0.

### loc_error_scale : *float*

For ALLoc/ICLLoc applying neighborhood sampling, this parameter determines the accuracy of initial locations. Ignored if *init_loc_dir* is not None. Default: 1.

### init_loc_dir : *str or None*

This parameter is for iterative search purpose.

For ALLoc/ICLLoc applying neighborhood sampling, this parameter determines how initial locations come from.

- If *init_loc_dir* is a directory, it should be the *exp_dir* parameter of a previous experiment whose *task* parameter is 'test'. The current experiment will use the inferred locations of the previous experiment as initial locations to perform iterative inference.
- If *init_loc_dir* is None, the ground truth locations (if *loc_error_scale* is 0) or coarse locations (if *loc_error_scale* is not 0) will be used as initial locations.

Default: None.

### model.exp_dir : *str*

The directory where the trained model is loaded from. This parameter should be the *exp_dir* parameter of a previous experiment whose *task* parameter is 'train'.

### model.epoch : *int or None*

- If *model.epoch* is an integer *E*, the model after *E* training epochs will be loaded from checkpoint.
- If *model.epoch* is None, the final model will be loaded.

---

## Examples

### Single-scenario learning and generalization

Train AD_CNN in O1 scenario.

```shell
python main.py --conf=./conf/train.yaml --mod exp_dir=./exp/o1_40k/ad_cnn/ \
    data_dirs=[./dataset_deepmimo/o1/] \
    scheme=ad_cnn num_train_data=40000 init_lr=0.001
```

Test the trained AD_CNN in O1 scenario.

```shell
python main.py --conf=./conf/test.yaml --mod exp_dir=./exp/o1_40k/ad_cnn/test/ \
    model.exp_dir=./exp/o1_40k/ad_cnn/ \
    data_dirs=[./dataset_deepmimo/o1/] \
    scheme=ad_cnn num_train_data=40000
```

### Cross-scenario learning and generalization

Train AD_CNN in O1 scenario.

```shell
python main.py --conf=./conf/train.yaml --mod exp_dir=./exp/o1_40k/ad_cnn/ \
    data_dirs=[./dataset_deepmimo/o1/] \
    scheme=ad_cnn num_train_data=40000 init_lr=0.001
```

Test the trained AD_CNN in O1B scenario.

```shell
python main.py --conf=./conf/test.yaml --mod exp_dir=./exp/o1_to_o1b_40k/ad_cnn_reuse/test_o1b/ \
    model.exp_dir=./exp/o1_40k/ad_cnn/ \
    data_dirs=[./dataset_deepmimo/o1b/] \
    scheme=ad_cnn num_train_data=40000
```

### Multi-scenario learning and generalization

Train AD_CNN in MO1 scenarios 1,2,4,5.

```shell
python main.py --conf=./conf/train.yaml --mod exp_dir=./exp/mo1_20k/ad_cnn/ \
    data_dirs=[./dataset_deepmimo/mo1_scen1/,./dataset_deepmimo/mo1_scen2/,./dataset_deepmimo/mo1_scen4/,./dataset_deepmimo/mo1_scen5/] \
    scheme=ad_cnn num_train_data=20000 init_lr=0.001
```

Test the trained AD_CNN in MO1 scenarios 1,2,3,4,5.

```shell
python main.py --conf=./conf/test.yaml --mod exp_dir=./exp/mo1_20k/ad_cnn/test/ \
    model.exp_dir=./exp/mo1_20k/ad_cnn/ \
    data_dirs=[./dataset_deepmimo/mo1_scen1/,./dataset_deepmimo/mo1_scen2/,./dataset_deepmimo/mo1_scen3/,./dataset_deepmimo/mo1_scen4/,./dataset_deepmimo/mo1_scen5/] \
    scheme=ad_cnn num_train_data=20000
```

### Transfer learning

Train AD_CNN in O1 scenario.

```shell
python main.py --conf=./conf/train.yaml --mod exp_dir=./exp/o1_40k/ad_cnn/ \
    data_dirs=[./dataset_deepmimo/o1/] \
    scheme=ad_cnn num_train_data=40000 init_lr=0.001
```

Load the AD_CNN trained in O1 scenario as initial model. Finetune in O1B scenario for 10 epochs and save checkpoint every epoch.

```shell
python main.py --conf=./conf/train.yaml --mod exp_dir=./exp/o1_to_o1b_40k/ad_cnn_finetune/ \
    finetune=True finetune_model.exp_dir=./exp/o1_40k/ad_cnn/ \
    data_dirs=[./dataset_deepmimo/o1b/] \
    scheme=ad_cnn num_train_data=40000 init_lr=0.0001 \
    num_epochs=10 save_epoch_inter=1
```

Load the AD_CNN finetuned after 5 epochs and test in O1 scenario (original scenario) and O1B scenario (new scenario), respectively.

```shell
python main.py --conf=./conf/test.yaml --mod exp_dir=./exp/o1_to_o1b_40k/ad_cnn_finetune/test_o1/epoch=5/ \
    model.exp_dir=./exp/o1_to_o1b_40k/ad_cnn_finetune/ model.epoch=5 \
    data_dirs=[./dataset_deepmimo/o1/] \
    scheme=ad_cnn num_train_data=40000
python main.py --conf=./conf/test.yaml --mod exp_dir=./exp/o1_to_o1b_40k/ad_cnn_finetune/test_o1b/epoch=5/ \
    model.exp_dir=./exp/o1_to_o1b_40k/ad_cnn_finetune/ model.epoch=5 \
    data_dirs=[./dataset_deepmimo/o1b/] \
    scheme=ad_cnn num_train_data=40000
```

### Multi-task learning

Train AD_CNN backbone in MO1 scenarios 1,2,4,5.

```shell
python main.py --conf=./conf/train.yaml --mod exp_dir=./exp/mo1_20k_multitask/ad_cnn_multitask_4scens/ \
    data_dirs=[./dataset_deepmimo/mo1_scen1/,./dataset_deepmimo/mo1_scen2/,./dataset_deepmimo/mo1_scen4/,./dataset_deepmimo/mo1_scen5/] \
    scheme=ad_cnn_multitask num_train_data=20000 init_lr=0.001
```

Load the trained AD_CNN backbone as initial model. Train scenario-specific head in MO1 scenario 3 for 50 epochs and save checkpoint every 5 epochs.

```shell
python main.py --conf=./conf/train.yaml --mod exp_dir=./exp/mo1_20k_multitask/ad_cnn_multitask_finetune_scen3/ \
    finetune=True finetune_model.exp_dir=./exp/mo1_20k_multitask/ad_cnn_multitask_4scens/ \
    data_dirs=[./dataset_deepmimo/mo1_scen3/] \
    scheme=ad_cnn_multitask num_train_data=20000 init_lr=0.001 \
    num_epochs=50 save_epoch_inter=5
```

Load the AD_CNN trained after 5 epochs and test in MO1 scenario 3.

```shell
python main.py --conf=./conf/test.yaml --mod exp_dir=./exp/mo1_20k_multitask/ad_cnn_multitask_finetune_scen3/test/epoch=5/ \
    model.exp_dir=./exp/mo1_20k_multitask/ad_cnn_multitask_finetune_scen3/ model.epoch=5 \
    data_dirs=[./dataset_deepmimo/mo1_scen3/] \
    scheme=ad_cnn_multitask num_train_data=20000
```

### Varying testing conditions

Train ALLoc in O1 scenario, using neighborhood sampling.

```shell
python main.py --conf=./conf/train.yaml --mod exp_dir=./exp/o1_40k/alloc_neighbor/ \
    data_dirs=[./dataset_deepmimo/o1/] \
    scheme=alloc select=neighbor num_train_data=40000 init_lr=0.0001
```

Test the trained ALLoc in O1 scenario with 32 neighboring samples.

```shell
python main.py --conf=./conf/test.yaml --mod exp_dir=./exp/o1_40k/alloc_neighbor/test_len_nei/len_nei=32/ \
    model.exp_dir=./exp/o1_40k/alloc_neighbor/ \
    data_dirs=[./dataset_deepmimo/o1/] \
    scheme=alloc select=neighbor num_train_data=40000 len_nei_test=32
```

Test the trained ALLoc in O1 scenario with noisy channels. Noise intensity is 0.1.

```shell
python main.py --conf=./conf/test.yaml --mod exp_dir=./exp/o1_40k/alloc_neighbor/test_noise_dev/noise_dev=0.1/ \
    model.exp_dir=./exp/o1_40k/alloc_neighbor/ \
    data_dirs=[./dataset_deepmimo/o1/] \
    scheme=alloc select=neighbor num_train_data=40000 noise_dev=0.1
```

Test the trained ALLoc in O1 scenario with inaccurate initial locations. The scale of initial location error is 4m.

```shell
python main.py --conf=./conf/test.yaml --mod exp_dir=./exp/o1_40k/alloc_neighbor/test_loc_error/loc_error_scale=4/ \
    model.exp_dir=./exp/o1_40k/alloc_neighbor/ \
    data_dirs=[./dataset_deepmimo/o1/] \
    scheme=alloc select=neighbor num_train_data=40000 loc_error_scale=4
```

### Iterative search

Train ALLoc in O1 scenario, using random sampling and neighborhood sampling, respectively.

```shell
python main.py --conf=./conf/train.yaml --mod exp_dir=./exp/o1_40k/alloc_random/ \
    data_dirs=[./dataset_deepmimo/o1/] \
    scheme=alloc select=random num_train_data=40000 init_lr=0.0001
python main.py --conf=./conf/train.yaml --mod exp_dir=./exp/o1_40k/alloc_neighbor/ \
    data_dirs=[./dataset_deepmimo/o1/] \
    scheme=alloc select=neighbor num_train_data=40000 init_lr=0.0001
```

Test ALLoc in O1 scenario, using random sampling, to obtain coarse locations.

```shell
python main.py --conf=./conf/test.yaml --mod exp_dir=./exp/o1_40k/alloc_random/test/ \
    model.exp_dir=./exp/o1_40k/alloc_random/ \
    data_dirs=[./dataset_deepmimo/o1/] \
    scheme=alloc select=random num_train_data=40000
```

Load the inferred locations by ALLoc using random sampling as initial locations. Test ALLoc using neighborhood sampling to perform iterative inference.

```shell
python main.py --conf=./conf/test.yaml --mod exp_dir=./exp/o1_40k/alloc_iterative/test/ \
    init_loc_dir=./exp/o1_40k/alloc_random/test/ \
    model.exp_dir=./exp/o1_40k/alloc_neighbor/ \
    data_dirs=[./dataset_deepmimo/o1/] \
    scheme=alloc select=neighbor num_train_data=40000
```