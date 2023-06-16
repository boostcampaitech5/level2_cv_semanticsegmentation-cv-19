import time

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import wandb
from trainer.base_trainer import BaseTrainer
from utils.util import MetricTracker, inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        save_dir,
        args,
        device,
        train_loader,
        val_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, save_dir, args)
        self.args = args
        self.device = device
        self.train_loader = train_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_loader)
        else:
            # iteration-based training
            self.train_loader = inf_loop(train_loader)
            self.len_epoch = len_epoch

        self.val_loader = val_loader
        self.do_validation = self.val_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = args.log_interval
        self.iters_to_accumulate = args.iters_to_accumulate

        # metric_fn들이 각 df의 index로 들어감
        self.train_metrics = MetricTracker("Loss", *[c.__class__.__name__ for c in self.criterion])  # DiceCoef
        self.valid_metrics = MetricTracker("Loss", *["DiceCoef"])
        # print(self.train_metrics._data.index)  # Index(['Loss', 'FocalLoss', 'DiceLoss', 'IoULoss', 'CombineLoss', 'DiceCoef'], dtype='object')

        self.scaler = GradScaler()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        start = time.time()
        self.model.train()
        self.train_metrics.reset()

        self.optimizer.zero_grad()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            total_loss = 0

            with autocast():
                output = self.model(data)

                # size가 달라진 경우 input_size와 같게 복원
                output_h, output_w = output.size(-2), output.size(-1)
                mask_h, mask_w = data.size(-2), data.size(-1)
                if output_h != mask_h or output_w != mask_w:
                    output = F.interpolate(output, size=(mask_h, mask_w), mode="bilinear")

                for loss_fn in self.criterion:  # [bce_with_logit, ...]
                    loss = loss_fn(output, target)
                    # print(f"{loss_fn} : ", loss)
                    self.train_metrics.update(loss_fn.__class__.__name__, loss.item())  # metric_fn마다 값 update
                    total_loss += loss / self.iters_to_accumulate

            self.scaler.scale(total_loss).backward()

            if (batch_idx + 1) % self.iters_to_accumulate == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # update loss value
            self.train_metrics.update("Loss", total_loss.item())

            if batch_idx % self.log_step == 0:
                print(f"Train Epoch: {epoch}/{self.args.epochs} {self._progress(batch_idx)} Loss: {total_loss.item():.6f}")
                log_dict = self.train_metrics.result()
                if self.args.is_wandb:
                    wandb.log({"Iter_train_" + k: v for k, v in log_dict.items()})  # logging per log_step (default=20)
            if batch_idx == self.len_epoch:
                break

        # 한 epoch 지난 이후, 반환할 결과 df 저장
        log = self.train_metrics.result()
        if self.args.is_wandb:
            wandb.log({"Epoch_Train_Loss": log["Loss"]})
        print("Train Epoch: {}, Loss: {:.6f}".format(epoch, self.train_metrics.result()["Loss"]))
        print(f"train time per epoch: {time.time()-start:.3f}s")
        print()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})  # val_log output도 넣어서 반환
            if self.args.is_wandb:
                wandb.log({"Epoch_val_" + k: v for k, v in val_log.items()})  # Epoch_val_Loss, Epoch_val_DiceCoef

        print()
        if self.lr_scheduler is not None:
            if self.args.is_wandb:
                wandb.log({"lr": self.optimizer.param_groups[0]["lr"]})
            self.lr_scheduler.step(log["val_Loss"])

        return log

    def _valid_epoch(self, epoch, thr=0.5):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        print("Validation Start!")
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                total_val_loss = 0

                output = self.model(data)

                output_h, output_w = output.size(-2), output.size(-1)
                mask_h, mask_w = target.size(-2), target.size(-1)

                # restore original size
                if output_h != mask_h or output_w != mask_w:
                    output = F.interpolate(output, size=(mask_h, mask_w), mode="bilinear")

                for loss_fn in self.criterion:  # [bce_with_logit, ...]
                    total_val_loss += loss_fn(output, target)

                output = torch.sigmoid(output)
                output = (output > thr).detach().cpu()
                target = target.detach().cpu()

                # update loss value
                self.valid_metrics.update("Loss", total_val_loss.item())
                for met in self.metric_ftns:
                    dice = met(output, target).mean(0)
                    self.valid_metrics.update(met.__class__.__name__, dice)
                    # print(dice)

        val_log_dict = self.valid_metrics.result()
        dice_coef = val_log_dict["DiceCoef"]
        val_log_dict["DiceCoef"] = val_log_dict["DiceCoef"].mean().item()
        for i in range(1, 30):
            val_log_dict[f"DiceCoef_class{i}"] = dice_coef[i - 1].item()

        return val_log_dict

    def _progress(self, batch_idx):
        base = "[{:>3d}/{}]"
        if hasattr(self.train_loader, "n_samples"):
            current = batch_idx * self.train_loader.batch_size
            total = self.train_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total)
