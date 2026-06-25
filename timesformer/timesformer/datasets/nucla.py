# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import DATASET_REGISTRY
from .smarthome import Smarthome


@DATASET_REGISTRY.register()
class Nucla(Smarthome):
    """
    NUCLA dataset loader.

    NUCLA currently uses the same split-file format and loading pipeline as
    Smarthome (`train.csv`/`val.csv`/`test.csv` with `path,label` rows).
    """

    pass
