import functools
import importlib
import json
import timeit
from types import SimpleNamespace
from typing import Any

import pydantic
import yaml
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch import nn


def timeit_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timer = timeit.Timer(lambda: func(*args, **kwargs))
        execution_time = timer.timeit(number=1)
        print(f"Function {func.__name__!r} executed in {execution_time:.4f} seconds")
        return func(*args, **kwargs)

    return wrapper


def load_module(config_dict: dict[Any, Any], classname: str) -> pydantic.BaseModel:

    module_name = config_dict.get("module", None)

    if module_name is not None:
        module = importlib.import_module(config_dict["module"])
        config_class = getattr(module, classname)
        config = config_class(**config_dict)
    else:
        module = importlib.import_module("__main__")
        config_class = getattr(module, classname)
        config = config_class(**config_dict)
    return config

def load_datamodule(config, dev: bool = False):
    module = importlib.import_module(config.module)
    train_dl, val_dl, test_dl, m, s = module.get_all_dataloaders(config, dev=dev)
    return SimpleNamespace(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        test_dataloader=test_dl,
        m=m,
        s=s,
    )


def load_model(config, model_path=None):
    module = importlib.import_module(config.module)
    if hasattr(module, "BaseLightningModel"):
            model_class = getattr(module, "BaseLightningModel")
    elif hasattr(module, "LightningModel"):
            model_class = getattr(module, "LightningModel")
    else:
            model_class = getattr(module, "Model")  # fallback


    if model_path:
        model = model_class.load_from_checkpoint(model_path)
    else:
        config_dict = json.loads(json.dumps(config, default=lambda s: vars(s)))
        model = model_class(config)
    return model

def load_logger(config):
    """
    Loads the logger.
    """
    module = importlib.import_module(config.module)
    return module.load_logger(config)

def load_config(path: str):
    with open(path, encoding="utf-8") as stream:
        cfg: dict[str, Any] = yaml.safe_load(stream)

    dataconfig = load_module(
        cfg["data"],
        classname="DataConfig",
    )
    modelconfig = load_module(
        cfg["model"],
        classname="ModelConfig",
    )
    trainerconfig = load_module(
        cfg["trainer"],
        classname="TrainerConfig",
    )
    loggerconfig = load_module(
        cfg["logger"],
        classname="LogConfig",
    )

    return (
        dataconfig,
        modelconfig,
        trainerconfig,
        loggerconfig,
    )
