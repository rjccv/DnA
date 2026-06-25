# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import csv
import os
import random
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import timesformer.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        self._label_to_id = None
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()

    def _split_names_for_mode(self, mode):
        split_names = [mode]
        if mode == "val":
            split_names.extend(["validate", "validation"])
        return split_names

    def _split_search_roots(self):
        split_root = self.cfg.DATA.PATH_TO_DATA_DIR
        return [
            os.path.join(split_root, "annotations"),
            split_root,
            os.path.join(split_root, "annotations2"),
        ]

    def _get_split_file_for_mode(self, mode):
        candidates = []
        for search_root in self._split_search_roots():
            for split_name in self._split_names_for_mode(mode):
                candidates.extend(
                    [
                        os.path.join(search_root, "{}.csv".format(split_name)),
                        os.path.join(search_root, "{}.txt".format(split_name)),
                        os.path.join(
                            search_root, "{}_manifest.txt".format(split_name)
                        ),
                    ]
                )
        for path in candidates:
            if PathManager.exists(path):
                return path
        raise FileNotFoundError(
            "Kinetics split file not found. Tried: {}".format(
                ", ".join(candidates)
            )
        )

    def _get_split_file(self):
        return self._get_split_file_for_mode(self.mode)

    def _resolve_video_path(self, path):
        if os.path.isabs(path):
            return path
        if self.cfg.DATA.PATH_PREFIX:
            return os.path.join(self.cfg.DATA.PATH_PREFIX, path)
        return os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, path)

    def _load_label_map_file(self):
        map_candidates = []
        for search_root in self._split_search_roots():
            map_candidates.extend(
                [
                    os.path.join(search_root, "kinetics_400_labels.csv"),
                    os.path.join(search_root, "kinetics400_labels.csv"),
                    os.path.join(search_root, "labels.csv"),
                ]
            )

        for path in map_candidates:
            if not PathManager.exists(path):
                continue
            label_to_id = {}
            with PathManager.open(path, "r") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                id_key = None
                name_key = None
                for key in ("id", "label_id", "class_id"):
                    if key in fieldnames:
                        id_key = key
                        break
                for key in ("name", "label", "class", "class_name"):
                    if key in fieldnames:
                        name_key = key
                        break

                if id_key is None or name_key is None:
                    logger.warning(
                        "Ignoring label map %s due to missing id/name columns",
                        path,
                    )
                    continue

                for row in reader:
                    if row.get(name_key, "").strip() == "":
                        continue
                    label_to_id[row[name_key].strip()] = int(row[id_key])

            if label_to_id:
                logger.info(
                    "Loaded Kinetics label map from %s (%d classes)",
                    path,
                    len(label_to_id),
                )
                return label_to_id
        return None

    def _infer_label_map_from_train_csv(self):
        train_file = self._get_split_file_for_mode("train")
        label_names = set()
        with PathManager.open(train_file, "r") as f:
            rows = csv.reader(f)
            for row in rows:
                row = [col.strip() for col in row if col.strip() != ""]
                if not row:
                    continue
                # Metadata style: label,youtube_id,time_start,time_end,split
                if len(row) >= 4:
                    if row[0].lower() == "label" and row[1].lower() == "youtube_id":
                        continue
                    label_names.add(row[0])
                    continue
                if len(row) == 2:
                    # Path + integer label split (already compatible, no map needed).
                    try:
                        int(row[1])
                        continue
                    except ValueError:
                        if row[0].lower() in {"path", "video_path", "filename"}:
                            continue
                        label_names.add(row[1])

        if not label_names:
            return None

        label_to_id = {name: idx for idx, name in enumerate(sorted(label_names))}
        logger.warning(
            "No explicit Kinetics label map found; inferred map from %s (%d classes).",
            train_file,
            len(label_to_id),
        )
        if len(label_to_id) != self.cfg.MODEL.NUM_CLASSES:
            logger.warning(
                "Inferred class count (%d) != cfg.MODEL.NUM_CLASSES (%d).",
                len(label_to_id),
                self.cfg.MODEL.NUM_CLASSES,
            )
        return label_to_id

    def _get_label_to_id(self):
        if self._label_to_id is not None:
            return self._label_to_id
        self._label_to_id = self._load_label_map_file()
        if self._label_to_id is None:
            self._label_to_id = self._infer_label_map_from_train_csv()
        return self._label_to_id

    def _label_from_token(self, label_token, line_no, path_to_file):
        label_token = label_token.strip()
        try:
            return int(label_token)
        except ValueError:
            label_map = self._get_label_to_id()
            if label_map is not None and label_token in label_map:
                return int(label_map[label_token])
            raise ValueError(
                "Invalid Kinetics label at {}:{} -> {} (no matching label map entry)".format(
                    path_to_file, line_no, label_token
                )
            )

    def _parse_split_row(self, row, line_no, path_to_file):
        row = [col.strip() for col in row if col.strip() != ""]
        if len(row) == 0:
            return None
        if len(row) == 1:
            sep = self.cfg.DATA.PATH_LABEL_SEPARATOR
            row = row[0].split() if sep.isspace() else row[0].split(sep)
            row = [col.strip() for col in row if col.strip() != ""]
        # Kinetics metadata CSV format:
        # label,youtube_id,time_start,time_end,split
        if len(row) >= 4:
            if (
                line_no == 1
                and row[0].lower() == "label"
                and row[1].lower() == "youtube_id"
            ):
                return None
            
            # Check for EgoExo4D format: path, label_name, is_exo, label_id
            if (row[0].endswith(".mp4") or row[0].endswith(".avi")) and row[3].isdigit():
                return row[0], int(row[3])

            label = self._label_from_token(row[0], line_no, path_to_file)
            youtube_id = row[1]
            try:
                start_sec = int(float(row[2]))
                end_sec = int(float(row[3]))
            except ValueError:
                raise ValueError(
                    "Invalid Kinetics timestamp at {}:{} -> {}".format(
                        path_to_file, line_no, row
                    )
                )
            split_dir = "val" if self.mode == "val" else self.mode
            video_path = "{}/{}_{:06d}_{:06d}.mp4".format(
                split_dir, youtube_id, start_sec, end_sec
            )
            return video_path, label

        if len(row) != 2:
            raise ValueError(
                "Invalid Kinetics annotation at {}:{} -> {}".format(
                    path_to_file, line_no, row
                )
            )
        video_path, label = row
        if line_no == 1 and (
            video_path.lower() in {"path", "video_path", "filename", "file"}
            or label.lower() in {"label", "label_id", "class", "class_id"}
        ):
            return None
        label = self._label_from_token(label, line_no, path_to_file)
        return video_path, label

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = self._get_split_file()

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with PathManager.open(path_to_file, "r") as f:
            if path_to_file.lower().endswith(".csv"):
                rows = csv.reader(f)
                parsed_rows = []
                for i, row in enumerate(rows):
                    parsed = self._parse_split_row(row, i + 1, path_to_file)
                    if parsed is not None:
                        parsed_rows.append(parsed)
            else:
                parsed_rows = []
                sep = self.cfg.DATA.PATH_LABEL_SEPARATOR
                for i, line in enumerate(f.read().splitlines()):
                    if not line.strip():
                        continue
                    parts = line.split() if sep.isspace() else line.split(sep)
                    parsed = self._parse_split_row(
                        parts, i + 1, path_to_file
                    )
                    if parsed is not None:
                        parsed_rows.append(parsed)

            for clip_idx, (path, label) in enumerate(parsed_rows):
                video_path = self._resolve_video_path(path)
                for idx in range(self._num_clips):
                    self._path_to_videos.append(video_path)
                    self._labels.append(label)
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self.mode, path_to_file
        )
        logger.info(
            "Constructing Kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                # if self.mode not in ["test"] and i_try > self._num_retries // 2:
                if self.mode not in ["test"] and i_try >= 1:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                # if self.mode not in ["test"] and i_try > self._num_retries // 2:
                if self.mode not in ["test"] and i_try >= 1:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue


            label = self._labels[index]

            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )

            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            frames = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )


            if self.cfg.MODEL.ARCH != 'vit':
                frames = utils.pack_pathway_output(self.cfg, frames)
            else:
                # Perform temporal sampling from the fast pathway.
                frames = torch.index_select(
                     frames,
                     1,
                     torch.linspace(
                         0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                     ).long(),
                )

            return frames, label, index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
