# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This script is used to get the mean/min/max/std estimates for each activation
of the model. The script operates on the directory that contains several
model checkpoints and an example input to the model. The script runs
every checkpoint on the input and gathers the statistics for each activation.
"""

import argparse
import logging
import sys
from typing import Any, List

import numpy as np
import torch
from fvcore.common.file_io import PathManager
from hydra.experimental import compose, initialize_config_module
from omegaconf import OmegaConf
from vissl.models import build_model
from vissl.utils.env import set_env_vars
from vissl.utils.hydra_config import convert_to_attrdict
from vissl.utils.io import makedir


# initiate the logger
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


class SSLHydraConfig(object):
    def __init__(self, overrides: List[Any] = None):
        self.overrides = []
        if overrides is not None and len(overrides) > 0:
            self.overrides.extend(overrides)
        self.overrides = [str(item) for item in self.overrides]
        with initialize_config_module(config_module="vissl.config"):
            cfg = compose(config_name="defaults", overrides=self.overrides)
        self.default_cfg = cfg

    @classmethod
    def from_configs(cls, config_files: List[Any] = None):
        return cls(config_files)

    def override(self, config_files: List[Any]):
        sys.argv = config_files
        cli_conf = OmegaConf.from_cli(config_files)
        self.default_cfg = OmegaConf.merge(self.default_cfg, cli_conf)


class SaveActivationOutput:
    def __init__(self):
        self.outputs = None

    def __call__(self, module, module_in, module_out):
        # if isinstance(module_out, list):
        #     # in case of multiple outputs or outputs as a list always. We ignore if
        #     # there are nested lists. To be fixed later.
        #     if not isinstance(module_out, list):
        #         self.outputs = module_out[0].detach()
        if isinstance(module_out, tuple) or isinstance(module_out, list):
            self.outputs = module_out[0].detach()
        else:
            self.outputs = module_out.detach()

    def clear(self):
        self.outputs = None


def collect_checkpoints(config, checkpoint_dir, use_iteration):
    output_files, all_iters = [], []
    all_files = PathManager.ls(checkpoint_dir)

    replace_prefix = "model_phase"
    # if we checkpoint at iterations too, we start from an iteration checkpoint
    # since that's latest than the phase end checkpoint. Sometimes, it's also
    # possible that there is no phase.
    if use_iteration:
        replace_prefix = "model_iteration"

    for f in all_files:
        # if we have the finished training, we pick the finished training file
        # the checkpoint is saved as "model_final_checkpoint". Otherwise, we pick
        # the latest phase checkpoint
        if replace_prefix in f:
            iter_num = int(f.replace(".torch", "").replace(replace_prefix, ""))
            all_iters.append(iter_num)
    if len(all_iters) > 0:
        all_iters.sort(reverse=False)

    for item in all_iters:
        output_files.append(f"{checkpoint_dir}/{replace_prefix}{item}.torch")
    return output_files


def collect_model_activations(
        input_sample, config, output_dir, checkpoint_path, model,
        use_iteration=False, whitelist=None):
    # checkpoint_files = collect_checkpoints(config, checkpoint_dir, use_iteration)
    # logger.info(checkpoint_files)
    sos, hooks = {}, {}
    for name, module in model.named_modules():
        in_module_name = True
        if whitelist:
            in_module_name = [s for s in whitelist if s in name]
            if not any(in_module_name):
                in_module_name = False
        if in_module_name:
            sos[name] = SaveActivationOutput()
            hooks[name] = module.register_forward_hook(sos[name])
            logging.info(f"Saving activations for {name}")

    activation_norms_m, activation_norms_s, activation_max = {}, {}, {}
    activations = {}
    logger.info(f"Checkpoint: {checkpoint_path}")
    with PathManager.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location=torch.device("cpu"))
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    state_dict = checkpoint.get("classy_state_dict")
    model.set_classy_state(state_dict["base_model"])
    model = model.half().eval()

    for name in sos.keys():
        sos[name].clear()

    with torch.no_grad():
        _ = model(input_sample)

    for name, v in sos.items():
        if v.outputs is not None:
            output = v.outputs.squeeze()
            # channels = output.shape[0]
            # dat = output.reshape(channels, -1)
            # norm = dat.norm(dim=1)
            # maxi = dat.max().item()
            #
            # if name not in activations.keys():
                # activations[name] = []
                # activation_norms_m[name] = []
                # activation_norms_s[name] = []
                # activation_max[name] = []
            activations[name] = output.cpu().numpy()
            # activation_norms_m[name].append(norm.mean().item())
            # activation_norms_s[name].append(norm.std().item())
            # activation_max[name].append(maxi)
# except Exception as e:
#     print(f"Exception: {e} for {checkpoint_path}")
#     np.save(f"{output_dir}/activations_norms_mean.npy", activation_norms_m)
#     np.save(f"{output_dir}/activations_norms_std.npy", activation_norms_s)
#     np.save(f"{output_dir}/activations_max.npy", activation_max)
    np.save(f"{output_dir}/activations.npy", activations)
    logger.info("Data saved. Done!")


def main():
    parser = argparse.ArgumentParser(description="Run on data and save "
                                                 "activations")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/",
        help="Output directory path where data will be saved",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/tmp/",
        help="Path to model checkpoint to load",
    )
    parser.add_argument(
        "--input_data_file", type=str, default="test", help="Input data file"
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help="For which date to sample. Must follow format: %Y-%m-%d",
    )
    parser.add_argument(
        "--whitelist",
        nargs="*",
        default=[],
        type=str,
        help="String(s) module's name must contain at least one of in order "
             "for its activations to be saved",
    )
    args = parser.parse_args()

    configs = args.configs.split(",")
    configs.extend(
        ["config.DISTRIBUTED.NUM_PROC_PER_NODE=1", "config.DISTRIBUTED.NUM_NODES=1"]
    )
    logger.info(f"Loading {configs}")

    cfg = SSLHydraConfig.from_configs(configs)
    _, config = convert_to_attrdict(cfg.default_cfg)
    set_env_vars(local_rank=0, node_id=0, cfg=config)
    if args.input_data_file == 'test':
        input_sample = torch.rand([2, 3, 224, 224]).cuda().half()
    else:
        input_sample = torch.load(args.input_data_file,
                                  map_location="cuda:0")["input"][0].half()

    model = build_model(config.MODEL, config.OPTIMIZER)
    model = model.cuda().half()

    output_dir = args.output_dir
    makedir(output_dir)
    collect_model_activations(input_sample, cfg, output_dir,
                              args.checkpoint_path, model,
                              whitelist=args.whitelist)


if __name__ == "__main__":
    main()

