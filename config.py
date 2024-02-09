import os

from experiment_config import Config, get_and_run_config_command

from src.modeling.util import MODEL_REGISTRY, TOKEN_POOLER_REGISTRY
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


@config.parameter(group="Model", default="bert-base-uncased", types=str)
def pretrained_model_name_or_path(val):  # noqa
    pass


@config.parameter(group="Model", default=False, types=bool)
def freeze_pretrained(val):
    pass


@config.parameter(group="Model", default="max", types=str)
def target_pool_fn(val):
    """
    Not used if model_name != 'stance-pooling-attention'
    """
    assert val in TOKEN_POOLER_REGISTRY.keys()


@config.parameter(group="Model", default="max", types=str)
def body_pool_fn(val):
    """
    Not used if model_name != 'stance-pooling'
    """
    assert val in TOKEN_POOLER_REGISTRY.keys()


@config.parameter(group="Model", default={}, types=dict)
def body_projection_fn_kwargs(val):
    """
    Not used if "attention" not in body_pool_fn.
    Keywords arguments passed to the projection function
    if using an attention pooling mechanism for body pooling.
    """
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
    assert config.Data.Encoder.pretrained_model_name_or_path.value == config.Model.pretrained_model_name_or_path.value  # noqa
    if "attention" in config.Model.model_name.value:
        assert "attention" in config.Model.body_pool_fn.value


if __name__ == "__main__":
    get_and_run_config_command(config)
