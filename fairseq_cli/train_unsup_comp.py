#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple
from fairseq.file_io import PathManager
from fairseq.dataclass.configs import CheckpointConfig
import logging
import ast
import collections

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.traincomp")

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from fairseq import checkpoint_utils, options, quantization_utils, tasks, utils
from fairseq.data import data_utils, iterators, indexed_dataset, MonolingualDataset
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
# from fairseq_cli.Comparable4 import Comparable
from fairseq_cli.Comparable_unsup import Comparable

def load_validation_data(data_path, src, tgt, src_dict, dataset_impl, split='valid', left_pad_source=True):
    def split_exists(split, src, tgt, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, src))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)
    
    if split_exists(split, src, tgt, data_path):
        prefix = os.path.join(data_path, "{}.{}-{}.".format(split, src, tgt))

    dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)

    return MonolingualDataset(dataset=dataset, sizes=dataset.sizes, src_vocab=src_dict, tgt_vocab=None, shuffle=False,add_eos_for_other_targets=False)


def get_valid_iterator(cfg, dataset, trainer, task, disable_iterator_cache=False):
    batch_iterator = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens_valid,
            max_sentences=cfg.dataset.batch_size_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.model.max_positions(),
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=trainer.data_parallel_world_size,
            shard_id=trainer.data_parallel_rank,
            num_workers=cfg.dataset.num_workers,
            # always pass a fixed "epoch" to keep validation data consistent
            # across training epochs
            epoch=1,
            data_buffer_size=cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
            skip_remainder_batch=False,
        )
    trainer.reset_dummy_batch(batch_iterator.first_batch)
    return batch_iterator

def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        print(f"convert namespace")
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)
    print(f"added user module")
    add_defaults(cfg)
    print(f"added defaults")

    if (
        distributed_utils.is_master(cfg.distributed_training)
        and "job_logging_cfg" in cfg
    ):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)
    # cfg.task.src_dict.add_symbol("<mask>")
    # cfg.task.tgt_dict.add_symbol("<mask>")

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg) #.model
    criterion = task.build_criterion(cfg.criterion)
    # generator = task.build_generator([model]) # SequenceGenerator object
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    # logger.info("generator: {}".format(generator.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(
                p.numel() for p in model.parameters() if not getattr(p, "expert", False)
            ),
            sum(
                p.numel()
                for p in model.parameters()
                if not getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(
                p.numel()
                for p in model.parameters()
                if getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    if not cfg.dataset.disable_validation:
        data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
        
        if cfg.comparable.comparable:
            paths = utils.split_paths(cfg.task.data)
            assert len(paths) > 0
            logger.info(f"paths: {paths}")
            src, tgt = cfg.task.source_lang, cfg.task.target_lang
            data_path = paths[0]
            logger.info(f"data_path: {data_path}")
            vaild_dataset = load_validation_data(data_path,src, tgt,src_dict=task.src_dict, dataset_impl='raw')

        elif cfg.dataset.combine_valid_subsets:
            task.load_dataset("valid", combine=True, epoch=1)
        else:
            for valid_sub_split in cfg.dataset.valid_subset.split(","):
                task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        logger.info("trainer")
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        logger.info("MegatronTrainer")
        trainer = MegatronTrainer(cfg, task, model, criterion)
    
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per device = {} and max sentences per device = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    # extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
    #     cfg.checkpoint,
    #     trainer,
    #     # don't cache epoch iterators for sharded datasets
    #     disable_iterator_cache=task.has_sharded_data("train"),
    # )
    extra_state, epoch = load_checkpoint(cfg, trainer)
    if cfg.common.tpu:
        import torch_xla.core.xla_model as xm

        xm.rendezvous("load_checkpoint")  # wait for all workers

    # Train until the learning rate gets too small
    max_epoch = cfg.optimization.max_epoch or math.inf
    # max_update = cfg.optimization.max_update or math.inf
    lr = trainer.get_lr()

    # TODO: a dry run on validation set to pin the memory
    # valid_subsets = cfg.dataset.valid_subset.split(",")
    if not cfg.dataset.disable_validation:
        logger.info('begin dry-run validation on valid subset')
        valid_itr = get_valid_iterator(cfg, vaild_dataset, trainer, task).next_epoch_itr(
                shuffle=False, set_dataset_epoch=False  # use a fixed valid set
            )
        # for subset in valid_subsets:
        #     logger.info('begin dry-run validation on "{}" subset'.format(subset))
        #     itr = trainer.get_valid_iterator(subset).next_epoch_itr(
        #         shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        #     )
        #     if cfg.common.tpu:
        #         itr = utils.tpu_data_loader(itr)
        #     for _ in itr:
        #         pass
    # TODO: end of dry run section

    train_meter = meters.StopwatchMeter()
    train_meter.start()
    
    if cfg.comparable.comparable:
        comp = Comparable(model, trainer, task, cfg)

        while epoch <= max_epoch: # _itr.next_epoch_idx
            if lr <= cfg.optimization.stop_min_lr:
                logger.info(
                    f"stopping training because current learning rate ({lr}) is smaller "
                    "than or equal to minimum learning rate "
                    f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
                )
                break

            # train for one epoch
            print(f"begin epoch")
            comp.task.begin_epoch(epoch, comp.trainer.get_model()) 
            #  _itr.next_epoch_idx
            # valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
            # if should_stop:
            #     break
            # print(f"epoch_itr.next_epoch_id: {epoch_itr.next_epoch_id}")
            # print(f"epoch_itr.epoch: {epoch_itr.epoch}")
            # Extract parallel data and train
            num_updates, end_of_epoch = comp.extract_and_train(cfg.comparable.comparable_data, epoch) #_itr.next_epoch_idx
            # num_updates, end_of_epoch = comp.unsupervised_training(cfg.comparable.comparable_data, epoch)
            max_update = cfg.optimization.max_update or math.inf
            should_stop = False

            if num_updates >= max_update:
                should_stop = True
                logger.info(
                    f"Stopping training due to "
                    f"num_updates: {num_updates} >= max_update: {max_update}"
                )

            training_time_hours = trainer.cumulative_training_time() / (60 * 60)
            if (
                cfg.optimization.stop_time_hours > 0
                and training_time_hours > cfg.optimization.stop_time_hours
            ):
                should_stop = True
                logger.info(
                    f"Stopping training due to "
                    f"cumulative_training_time: {training_time_hours} > "
                    f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
                )

            do_save = (
                (end_of_epoch and epoch % cfg.checkpoint.save_interval == 0) 
                or should_stop
                or (
                    cfg.checkpoint.save_interval_updates > 0
                    and num_updates > 0
                    and num_updates % cfg.checkpoint.save_interval_updates == 0
                    and num_updates >= cfg.dataset.validate_after_updates
                )
            )
            do_validate = (
                (
                    (not end_of_epoch and do_save)  # validate during mid-epoch saves
                    or (end_of_epoch and epoch % cfg.dataset.validate_interval == 0)
                    or should_stop
                    or (
                        cfg.dataset.validate_interval_updates > 0
                        and num_updates > 0
                        and num_updates % cfg.dataset.validate_interval_updates == 0
                    )
                )
                and not cfg.dataset.disable_validation
                and num_updates >= cfg.dataset.validate_after_updates
            )
            # epoch_itr.
            # Validate
            valid_losses = [None] 
            if do_validate:
                valid_losses = comp.validate(epoch, valid_itr)

                valid_itr = get_valid_iterator(cfg, vaild_dataset, trainer, task).next_epoch_itr(
                shuffle=False, set_dataset_epoch=False  # use a fixed valid set
            )
            # _itr.next_epoch_idx
            # if (not cfg.dataset.disable_validation
            #     and cfg.checkpoint.save_interval_updates > 0
            #     and num_updates % cfg.checkpoint.save_interval_updates == 0
            #     and num_updates > 0
            # ):
            #    valid_losses = comp.validate(epoch_itr.next_epoch_idx, valid_subsets)
            # else:
            #     valid_losses = [None] 

            should_stop |= should_stop_early(cfg, valid_losses[0])

            # Save checkpoint
            if do_save or should_stop:
                cp_path = save_checkpoint(
                    cfg.checkpoint, trainer, epoch, valid_losses[0]
                )
                if cp_path is not None and hasattr(task, "post_save"):
                    task.post_save(cp_path, num_updates)
            
            if should_stop:
                break

            # only use first validation loss to update the learning rate
            lr = trainer.lr_step(epoch, valid_losses[0])
            epoch += 1

            # epoch_itr = trainer.get_train_iterator(
            #     epoch,
            #     # sharded data: get train iterator for next epoch
            #     load_dataset=task.has_sharded_data("train"),
            #     # don't cache epoch iterators for sharded datasets
            #     disable_iterator_cache=task.has_sharded_data("train"),
            # )
        train_meter.stop()
        logger.info("done training in {:.1f} seconds".format(train_meter.sum))

        # ioPath implementation to wait for all asynchronous file writes to complete.
        if cfg.checkpoint.write_checkpoints_asynchronously:
            logger.info(
                "ioPath PathManager waiting for all asynchronous checkpoint "
                "writes to finish."
            )
            PathManager.async_close()
            logger.info("ioPath PathManager finished waiting.")

def load_checkpoint(cfg, trainer, **passthrough_args):
    """
    Load a checkpoint and restore the training iterator.

    *passthrough_args* will be passed through to
    ``trainer.get_train_iterator``.
    """
    # only one worker should attempt to create the required dir
    reset_optimizer = cfg.checkpoint.reset_optimizer
    reset_lr_scheduler = cfg.checkpoint.reset_lr_scheduler
    # print(f"cfg.optimizer_overrides: {cfg.optimizer_overrides}")
    optimizer_overrides = ast.literal_eval(cfg.checkpoint.optimizer_overrides)
    reset_meters = cfg.checkpoint.reset_meters
    reset_dataloader = cfg.checkpoint.reset_dataloader

    if cfg.distributed_training.distributed_rank == 0:
        print(f"cfg.checkpoint.save_dir: {cfg.checkpoint.save_dir}")
        os.makedirs(cfg.checkpoint.save_dir, exist_ok=True)

    if cfg.checkpoint.finetune_from_model is not None and (
        reset_optimizer or reset_lr_scheduler or reset_meters or reset_dataloader
    ):
        raise ValueError(
            "--finetune-from-model can not be set together with either --reset-optimizer"
            " or reset_lr_scheduler or reset_meters or reset_dataloader"
        )
    suffix = trainer.checkpoint_suffix

    if cfg.checkpoint.restore_file == "checkpoint_last.pt":
        checkpoint_path = os.path.join(cfg.checkpoint.save_dir, "checkpoint_last{}.pt".format(suffix))
        first_launch = not PathManager.exists(checkpoint_path)
        if first_launch and getattr(cfg.checkpoint, "continue_once", None) is not None:
            checkpoint_path = cfg.checkpoint.continue_once
        elif cfg.checkpoint.finetune_from_model is not None and first_launch:
            # if there is no last checkpoint to restore, start the finetune from pretrained model
            # else just use usual logic to load checkpoint, e.g. restart from last checkpoint and etc.
            if PathManager.exists(cfg.checkpoint.finetune_from_model):
                checkpoint_path = cfg.checkpoint.finetune_from_model
                reset_optimizer = True
                reset_lr_scheduler = True
                reset_meters = True
                reset_dataloader = True
                logger.info(
                    f"loading pretrained model from {checkpoint_path}: "
                    "optimizer, lr scheduler, meters, dataloader will be reset"
                )
            else:
                raise ValueError(
                    f"--finetune-from-model {cfg.finetune_from_model} does not exist"
                )
    elif suffix is not None:
        checkpoint_path = cfg.checkpoint.restore_file.replace(".pt", suffix + ".pt")
    else:
        checkpoint_path = os.path.join(cfg.checkpoint.save_dir, cfg.checkpoint.restore_file)

    if cfg.checkpoint.restore_file != "checkpoint_last.pt" and cfg.checkpoint.finetune_from_model:
        raise ValueError(
            "--finetune-from-model and --restore-file (non-default value) "
            "can not be specified together: " + str(cfg)
        )

    extra_state = trainer.load_checkpoint(
        checkpoint_path,
        reset_optimizer,
        reset_lr_scheduler,
        optimizer_overrides,
        reset_meters=reset_meters,
    )

    # if (
    #     extra_state is not None
    #     and "best" in extra_state
    #     and not args.reset_optimizer
    #     and not args.reset_meters
    # ):
    #     save_checkpoint.best = extra_state["best"]

    if (
        extra_state is not None
        and "best" in extra_state
        and not reset_optimizer
        and not reset_meters
    ):
        save_checkpoint.best = extra_state["best"]

    if extra_state is not None and not reset_dataloader:
        # restore iterator from checkpoint
        itr_state = extra_state["train_iterator"]
        # epoch_itr = trainer.get_train_iterator(
        #     epoch=itr_state["epoch"], load_dataset=False, **passthrough_args
        # )
        epoch = extra_state["train_iterator"]["epoch"] + 1
        # epoch_itr.load_state_dict(itr_state)
    else:
        epoch = 1
        # epoch_itr = trainer.get_train_iterator(
        #     epoch=1, load_dataset=False, **passthrough_args
        # )

    trainer.lr_step(epoch)
    # trainer.lr_step(epoch_itr.epoch)

    return extra_state, epoch
    # return extra_state, epoch_itr


def save_checkpoint(cfg: CheckpointConfig, trainer, epoch, val_loss):
    from fairseq import meters

    # only one worker should attempt to create the required dir
    if trainer.data_parallel_rank == 0:
        os.makedirs(cfg.save_dir, exist_ok=True)

    prev_best = getattr(save_checkpoint, "best", val_loss)
    if val_loss is not None:
        best_function = max if cfg.maximize_best_checkpoint_metric else min
        save_checkpoint.best = best_function(val_loss, prev_best)

    if cfg.no_save:
        return None

    trainer.consolidate_optimizer()  # TODO(SS): do we need this if no_save_optimizer_state

    if not trainer.should_save_checkpoint_on_current_rank:
        if trainer.always_call_state_dict_during_save_checkpoint:
            trainer.state_dict()
        return None

    write_timer = meters.StopwatchMeter()
    write_timer.start()

    # epoch = epoch_itr.epoch
    # end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    logger.info(f"Preparing to save checkpoint for epoch {epoch} @ {updates} updates")

    def is_better(a, b):
        return a >= b if cfg.maximize_best_checkpoint_metric else a <= b

    suffix = trainer.checkpoint_suffix
    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds["checkpoint{}{}.pt".format(epoch, suffix)] = (
        # end_of_epoch and 
        not cfg.no_epoch_checkpoints and epoch % cfg.save_interval == 0
    )
    checkpoint_conds["checkpoint_{}_{}{}.pt".format(epoch, updates, suffix)] = (
        # not end_of_epoch and 
        cfg.save_interval_updates > 0
        and updates % cfg.save_interval_updates == 0
    )
    checkpoint_conds["checkpoint_best{}.pt".format(suffix)] = val_loss is not None and (
        not hasattr(save_checkpoint, "best")
        or is_better(val_loss, save_checkpoint.best)
    )
    if val_loss is not None and cfg.keep_best_checkpoints > 0:
        worst_best = getattr(save_checkpoint, "best", None)
        chkpts = checkpoint_utils.checkpoint_paths(
            cfg.save_dir,
            pattern=r"checkpoint\.best_{}_(\d+\.?\d*){}\.pt".format(
                cfg.best_checkpoint_metric, suffix
            ),
        )
        if len(chkpts) > 0:
            p = chkpts[-1] if cfg.maximize_best_checkpoint_metric else chkpts[0]
            worst_best = float(p.rsplit("_")[-1].replace("{}.pt".format(suffix), ""))
        # add random digits to resolve ties
        with data_utils.numpy_seed(epoch, updates, val_loss):
            rand_sfx = np.random.randint(0, cfg.keep_best_checkpoints)

        checkpoint_conds[
            "checkpoint.best_{}_{:.3f}{}{}.pt".format(
                cfg.best_checkpoint_metric, val_loss, rand_sfx, suffix
            )
        ] = worst_best is None or is_better(val_loss, worst_best)
    checkpoint_conds[
        "checkpoint_last{}.pt".format(suffix)
    ] = not cfg.no_last_checkpoints

    extra_state = {"train_iterator": {"epoch": epoch}, "val_loss": val_loss}
    if hasattr(save_checkpoint, "best"):
        extra_state.update({"best": save_checkpoint.best})

    checkpoints = [
        os.path.join(cfg.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ]
    saved_cp = None
    if len(checkpoints) > 0 and trainer.should_save_checkpoint_on_current_rank:
        saved_cp = trainer.save_checkpoint(checkpoints[0], extra_state)
        for cp in checkpoints[1:]:
            if cfg.write_checkpoints_asynchronously:
                # TODO[ioPath]: Need to implement a delayed asynchronous
                # file copying/moving feature.
                logger.warning(
                    f"ioPath is not copying {checkpoints[0]} to {cp} "
                    "since async write mode is on."
                )
            else:
                assert PathManager.copy(
                    checkpoints[0], cp, overwrite=True
                ), f"Failed to copy {checkpoints[0]} to {cp}"

        write_timer.stop()
        logger.info(
            "Saved checkpoint {} (epoch {} @ {} updates, score {}) (writing took {} seconds)".format(
                checkpoints[0], epoch, updates, val_loss, write_timer.sum
            )
        )

    # if (
    #     # not end_of_epoch and 
    #     cfg.keep_interval_updates > 0
    #     and trainer.should_save_checkpoint_on_current_rank
    # ):
    #     # remove old checkpoints; checkpoints are sorted in descending order
    #     if cfg.keep_interval_updates_pattern == -1:
    #         checkpoints = checkpoint_paths(
    #             cfg.save_dir, pattern=r"checkpoint_\d+_(\d+){}\.pt".format(suffix)
    #         )
    #     else:
    #         checkpoints = checkpoint_paths(
    #             cfg.save_dir,
    #             pattern=r"checkpoint_\d+_(\d+){}\.pt".format(suffix),
    #             keep_match=True,
    #         )
    #         checkpoints = [
    #             x[0]
    #             for x in checkpoints
    #             if x[1] % cfg.keep_interval_updates_pattern != 0
    #         ]

    #     for old_chk in checkpoints[cfg.keep_interval_updates :]:
    #         if os.path.lexists(old_chk):
    #             os.remove(old_chk)
    #         elif PathManager.exists(old_chk):
    #             PathManager.rm(old_chk)

    if cfg.keep_last_epochs > 0 and trainer.should_save_checkpoint_on_current_rank:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_utils.checkpoint_paths(
            cfg.save_dir, pattern=r"checkpoint(\d+){}\.pt".format(suffix)
        )
        for old_chk in checkpoints[cfg.keep_last_epochs :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
            elif PathManager.exists(old_chk):
                PathManager.rm(old_chk)

    if cfg.keep_best_checkpoints > 0 and trainer.should_save_checkpoint_on_current_rank:
        # only keep the best N checkpoints according to validation metric
        checkpoints = checkpoint_utils.checkpoint_paths(
            cfg.save_dir,
            pattern=r"checkpoint\.best_{}_(\d+\.?\d*){}\.pt".format(
                cfg.best_checkpoint_metric, suffix
            ),
        )
        if not cfg.maximize_best_checkpoint_metric:
            checkpoints = checkpoints[::-1]
        for old_chk in checkpoints[cfg.keep_best_checkpoints :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
            elif PathManager.exists(old_chk):
                PathManager.rm(old_chk)

    return saved_cp


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(
        itr,
        update_freq,
        skip_remainder_batch=cfg.optimization.skip_remainder_batch,
    )
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        aim_repo=(
            cfg.common.aim_repo
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_run_hash=(
            cfg.common.aim_run_hash
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (
            (not end_of_epoch and do_save)  # validate during mid-epoch saves
            or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
            or should_stop
            or (
                cfg.dataset.validate_interval_updates > 0
                and num_updates > 0
                and num_updates % cfg.dataset.validate_interval_updates == 0
            )
        )
        and not cfg.dataset.disable_validation
        and num_updates >= cfg.dataset.validate_after_updates
    )

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        cp_path = checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )
        if cp_path is not None and hasattr(task, "post_save"):
            task.post_save(cp_path, num_updates)

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset_idx, subset in enumerate(subsets):
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            aim_repo=(
                cfg.common.aim_repo
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            aim_run_hash=(
                cfg.common.aim_run_hash
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if (
                    cfg.dataset.max_valid_steps is not None
                    and i > cfg.dataset.max_valid_steps
                ):
                    break
                trainer.valid_step(sample)

        # log validation stats
        # only tracking the best metric on the 1st validation subset
        tracking_best = subset_idx == 0
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values(), tracking_best)

        if hasattr(task, "post_validate"):
            task.post_validate(trainer.get_model(), stats, agg)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(
    cfg: DictConfig,
    trainer: Trainer,
    stats: Dict[str, Any],
    tracking_best: bool,
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if tracking_best and hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    print(f"get parser")
    parser = options.get_training_parser()
    print(f"parser: {parser}")
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    print(f"args: {args}")

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(
            f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}"
        )

    if cfg.common.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()
