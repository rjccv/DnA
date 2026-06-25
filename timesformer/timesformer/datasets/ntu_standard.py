# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import timesformer.utils.logging as logging

from .build import DATASET_REGISTRY
from .smarthome import Smarthome

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ntu(Smarthome):
    """
    NTU dataset loader using the standard 4-tuple classification contract:
    `(frames, label, index, meta)`.

    It reuses Smarthome CSV parsing (`path,label` or `path,pose,label`) and
    split discovery (`train.csv`/`val.csv`/`test.csv`).
    """

    def _construct_loader(self):
        super()._construct_loader()
        self._normalize_labels()

    def _normalize_labels(self):
        if not self._labels:
            return

        unique_labels = sorted(set(self._labels))
        min_label, max_label = unique_labels[0], unique_labels[-1]

        if min_label < 0:
            raise ValueError(
                "NTU labels must be non-negative. Found min label {}.".format(
                    min_label
                )
            )

        # Common NTU annotation convention is 1-based class ids.
        if min_label == 1 and 0 not in unique_labels:
            self._labels = [label - 1 for label in self._labels]
            unique_labels = sorted(set(self._labels))
            max_label = unique_labels[-1]
            logger.info("Converted NTU labels from 1-based to 0-based indices.")

        if max_label >= self.cfg.MODEL.NUM_CLASSES:
            raise ValueError(
                (
                    "NTU label id {} is out of range for MODEL.NUM_CLASSES={} "
                    "(valid ids: 0..{}). Update MODEL.NUM_CLASSES or fix label "
                    "indexing in split CSVs."
                ).format(
                    max_label,
                    self.cfg.MODEL.NUM_CLASSES,
                    self.cfg.MODEL.NUM_CLASSES - 1,
                )
            )

        if len(unique_labels) != self.cfg.MODEL.NUM_CLASSES:
            logger.warning(
                "NTU split has %d unique labels while MODEL.NUM_CLASSES=%d.",
                len(unique_labels),
                self.cfg.MODEL.NUM_CLASSES,
            )
