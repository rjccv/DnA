#!/usr/bin/env python3


from timesformer.utils.misc import launch_job
from timesformer.utils.parser import load_config, parse_args
from tools.test_net import test


def main():
    args = parse_args()
    cfg = load_config(args)

    # Force test-only regardless of config defaults.
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True

    launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
