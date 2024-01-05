import os

from experiment_config import Config, get_and_run_config_command

from src.modeling.util import MODEL_REGISTRY
from src.data.util import ENCODER_REGISTRY


config = Config("StanceDetectionConfig")


@config.parameter(group="Experiment", types=str)
def name(val):
    assert val != ''


@config.parameter(group="Experiment", types=str)
def description(val):
    assert val != ''


@config.parameter(group="Experiment", types=str)
def logdir(val):
    pass


@config.parameter(group="Experiment", default=0, types=int)
def random_seed(val):
    pass


@config.parameter(group="Experiment", default=0, types=int)
def version(val):
    assert val >= 0


@config.parameter(group="Data", types=str)
def datadir(val):
    assert os.path.isdir(val)


@config.parameter(group="Data", default=-1, types=int)
def num_examples(val):
    assert val == -1 or val > 0


@config.parameter(group="Data", default="all", types=(str, list))
def tasks_to_load(val):
    """
    Can be either "all" or a list of strings specifying the tasks.
    """
    if isinstance(val, list):
        for item in val:
            assert isinstance(item, str)


@config.parameter(group="Data.Encoder", default="default", types=(type(None), str))  # noqa
def encoder_type(val):
    assert val is None or val in ENCODER_REGISTRY.keys()


@config.parameter(group="Data.Encoder", default="bert-base-uncased", types=str)  # noqa
def pretrained_model_name_or_path(val):
    pass


@config.parameter(group="Data.Encoder", default=256, types=int)
def max_seq_length(val):
    assert val > 0


@config.parameter(group="Model", default="default", types=str)
def model_name(val):
    assert val in MODEL_REGISTRY.keys()


@config.parameter(group="Model", default="bert-base-uncased", types=str)  # noqa
def pretrained_model_name_or_path(val):
    pass


@config.parameter(group="Model", default=0.0, types=float)
def dropout_prob(val):
    assert 0.0 <= val <= 1.0


@config.parameter(group="Training", default=1, types=int)
def epochs(val):
    assert val > 0


@config.parameter(group="Training", default=32, types=int)
def batch_size(val):
    assert val > 0


@config.parameter(group="Training", default=2e-5, types=float)
def learn_rate(val):
    assert val > 0.0


@config.parameter(group="Training", default=0.0, types=float)
def weight_decay(val):
    assert val >= 0.0


@config.parameter(group="Training", default=1, types=int)
def accumulate_grad_batches(val):
    assert val > 0


@config.on_load
def validate_parameters():
    assert config.Data.Encoder.pretrained_model_name_or_path == config.Model.pretrained_model_name_or_path  # noqa


if __name__ == "__main__":
    get_and_run_config_command(config)
