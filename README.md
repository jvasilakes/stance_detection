# Stance Detection

This project depends upon `experiment_config`: [https://github.com/jvasilakes/experiment-config]


## Data Preparation

First, download the datasets.

```
bash download.sh
```

This will create `data/{ARC,FNC}/raw`.


Next, preprocess the data, removing duplicate entries and creating the validation splits.
This will also print out a markdown formatted table of summary statistics.

```
python preprocess.py data/ARC/raw data/ARC/preprocessed/
python preprocess.py data/FNC/raw data/FNC/preprocessed/
```

## Model Training 

You will need to create a config file to specify hyperparameters for your experiment.

```
mkdir configs/
python config.py new configs/myconfig.yaml
```

Edit the config file as you desire. The `datadir` should be one of the `data/{ARC,FNC}/preprocessed`
directories you created above. Once you've created it, you can validate it with the following command.

```
python config.py print configs/myconfig.yaml
```

If this prints your config file without errors, you are ready to train!

```
python run.py train configs/myconfig.yaml
```

Relevant data regarding this experiment will be saved at `{logdir}/{name}/version_{version}/seed_{random_seed}` according to the values in the config file. This directory will be referred to as `experiment_logdir`.

## Evaluation

You can run a simple evaluation like so

```
python run.py validate --split val configs/myconfig.yaml
```

This will load the model saved during training and evaluate it on the validation set.


You can also save the model predictions for later evaluation with

```
python run.py predict --split val configs/myconfig.yaml
```

This will save the model predictions as `{experiment_logdir}/predictions/val/{task}.yaml`, where `{task}` is one of the tasks specified under `tasks_to_load` in the config file.
