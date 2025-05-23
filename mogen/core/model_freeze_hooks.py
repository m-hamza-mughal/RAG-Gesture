from typing import Optional
import os
import json
import time

from mmcv.runner import Hook
from mmcv.runner import Runner

from mmcv.runner import HOOKS


@HOOKS.register_module()
class VAE_FreezeHook(Hook):
    """put vae into eval mode and freeze the parameters."""

    def __init__(self, freeze: Optional[bool] = True):
        self.freeze = freeze

    def before_train_epoch(self, runner: Runner):
        # breakpoint()
        if self.freeze:
            # breakpoint() # check new implementation
            if runner.model.module.model.gesture_rep_encoder is not None:
                runner.model.module.model.gesture_rep_encoder.upper_vae.eval()
                runner.model.module.model.gesture_rep_encoder.face_vae.eval()
                runner.model.module.model.gesture_rep_encoder.hands_vae.eval()
                runner.model.module.model.gesture_rep_encoder.lowertrans_vae.eval()


    # def after_train_epoch(self, runner: Runner):s
    #     if self.freeze:
    #         breakpoint()


@HOOKS.register_module()
class BERT_FreezeHook(Hook):
    """put bert into eval mode and freeze the parameters."""

    def __init__(self, freeze: Optional[bool] = True):
        self.freeze = freeze

    def before_train_epoch(self, runner: Runner):
        if self.freeze:
            runner.model.module.model.bert.eval()


@HOOKS.register_module()
class DatabaseSaveHook(Hook):
    """Save the database to a files after each epoch."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def before_run(self, runner: Runner):
        # breakpoint()
        # print("before_run")
        if runner.model.module.model.database is None:
            return
        if runner.model.module.model.database.training:
            if os.path.exists(self.save_dir + "/train_indexes.json"):
                with open(
                    self.save_dir + "/train_indexes.json", "r", encoding="utf-8"
                ) as f:
                    runner.model.module.model.database.train_indexes = json.load(f)
                print("Loaded train indexes")
            if os.path.exists(self.save_dir + "/train_dbounds.json"):
                with open(
                    self.save_dir + "/train_dbounds.json", "r", encoding="utf-8"
                ) as f:
                    runner.model.module.model.database.train_dbounds = json.load(f)
                print("Loaded train dbounds")
            if os.path.exists(self.save_dir + "/train_qbounds.json"):
                with open(
                    self.save_dir + "/train_qbounds.json", "r", encoding="utf-8"
                ) as f:
                    runner.model.module.model.database.train_qbounds = json.load(f)
                print("Loaded train qbounds")

        if not runner.model.module.model.database.training:

            if os.path.exists(self.save_dir + "/test_indexes.json"):
                with open(
                    self.save_dir + "/test_indexes.json", "r", encoding="utf-8"
                ) as f:
                    runner.model.module.model.database.test_indexes = json.load(f)
                print("Loaded test indexes")
            if os.path.exists(self.save_dir + "/test_dbounds.json"):
                with open(
                    self.save_dir + "/test_dbounds.json", "r", encoding="utf-8"
                ) as f:
                    runner.model.module.model.database.test_dbounds = json.load(f)
                print("Loaded test dbounds")
            if os.path.exists(self.save_dir + "/test_qbounds.json"):
                with open(
                    self.save_dir + "/test_qbounds.json", "r", encoding="utf-8"
                ) as f:
                    runner.model.module.model.database.test_qbounds = json.load(f)
                print("Loaded test qbounds")

    def save_database(self, runner: Runner):
        """
        Save the database to a file.
        """
        if runner.model.module.model.database is None:
            return
        start = time.time()
        if runner.model.module.model.database.training:
            print("Saving train indexes")
            with open(
                self.save_dir + "/train_indexes.json", "w", encoding="utf-8"
            ) as f:
                json.dump(
                    runner.model.module.model.database.train_indexes,
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
            with open(
                self.save_dir + "/train_dbounds.json", "w", encoding="utf-8"
            ) as f:
                json.dump(
                    runner.model.module.model.database.train_dbounds,
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
            with open(
                self.save_dir + "/train_qbounds.json", "w", encoding="utf-8"
            ) as f:
                json.dump(
                    runner.model.module.model.database.train_qbounds,
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

        if not runner.model.module.model.database.training:
            print("Saving test indexes")
            with open(self.save_dir + "/test_indexes.json", "w", encoding="utf-8") as f:
                json.dump(
                    runner.model.module.model.database.test_indexes,
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
            with open(self.save_dir + "/test_dbounds.json", "w", encoding="utf-8") as f:
                json.dump(
                    runner.model.module.model.database.test_dbounds,
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
            with open(self.save_dir + "/test_qbounds.json", "w", encoding="utf-8") as f:
                json.dump(
                    runner.model.module.model.database.test_qbounds,
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

        print("Saved database in", time.time() - start, "seconds")

    def after_train_iter(self, runner: Runner):
        return

    def after_train_epoch(self, runner: Runner):
        

        # only save after the first epoch
        if runner.epoch == 0 and runner.model.module.model.database is not None:
            print("after_first_train_epoch")
            assert len(runner.model.module.model.database.train_indexes) != 0
            self.save_database(runner)

    def after_test_epoch(self, runner: Runner):
        # breakpoint()
        print("after_test_epoch")
        assert len(runner.model.module.model.database.test_indexes) != 0
        self.save_database(runner)
