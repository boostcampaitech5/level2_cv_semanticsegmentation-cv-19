from abc import abstractmethod

import numpy as np
import torch


class BaseTrainer:
    """
    Base class for all trainers, Describe the learning process of each epoch
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, save_dir, args):
        self.args = args

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.epochs = args.epochs
        self.save_dir = save_dir

        # configuration to monitor model performance and save best
        self.early_stop = args.early_stop

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        patient = 0
        best_val_dice = 0
        best_val_loss = np.inf
        print("Training Start")
        for epoch in range(1, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            print("--- Log print ---")
            for key, value in log.items():
                if str(key) == "epoch":
                    print(f"{str(key):}: {value}")
                else:
                    print(f"{str(key):}: {value:.5f}")

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if log["val_DiceCoef"] > best_val_dice:
                print(f"New best model for val DiceCoef : {log['val_DiceCoef']:.4f}%! saving the best model..")
                torch.save(self.model.state_dict(), f"{self.save_dir}/best_epoch.pth")
                best_val_dice = log["val_DiceCoef"]
                best_val_loss = log["val_Loss"]
                patient = 0
            else:
                patient += 1

            torch.save(self.model.state_dict(), f"{self.save_dir}/latest.pth")
            print(
                f"[Val] DiceCoef : {log['val_DiceCoef']:.4f}, Loss: {log['val_Loss']:.4f} || best DiceCoef : {best_val_dice:.4f}, best Loss: {best_val_loss:.4f} || patient: {patient}\n"
            )

            if self.early_stop < patient:
                print(f"Early stopping is triggerd at epoch {epoch}")
                print("Best performance:")
                print(
                    f"[Val] DiceCoef : {log['val_DiceCoef']:.4f}, Loss: {log['val_Loss']:.4f} || best DiceCoef : {best_val_dice:.4f}, best Loss: {best_val_loss:.4f}"
                )
                break

    # Not used at [230606] modified code
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "args": self.args,
        }
        filename = f"{self.save_dir}/latest.pth"
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))

        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that of "
                "checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint["config"]["optimizer"]["type"] != self.config["optimizer"]["type"]:
            self.logger.warning(
                "Warning: Optimizer type given in config file is different from that of checkpoint. "
                "Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
