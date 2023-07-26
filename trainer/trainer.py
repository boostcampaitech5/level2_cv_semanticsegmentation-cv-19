import time

import torch
import torch.nn.functional as F
import wandb
from torch.cuda.amp import GradScaler, autocast
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

        # metric_fn들이 각 df의 index로 들어감
        if args.multi_task:
            self.train_metrics = MetricTracker("Loss", *[c for c in self.args.criterion])  # DiceCoef
            self.log_metrics = MetricTracker(*[f"{c}_var" for c in self.args.criterion])
        else:
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
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # dt = data[0].numpy()  # .transpose(1, 2, 0)
            # print(dt.shape)
            # print(dt.max(), dt.min())
            # cv2.imwrite(f"./example/elastic3/ex_{batch_idx}.png", dt[0] * 255)
            # print(data.shape, target.shape)
            data, target = data.to(self.device), target.to(self.device)
            total_loss = 0

            self.optimizer.zero_grad()

            with autocast():
                output = self.model(data)

                # size가 달라진 경우 input_size와 같게 복원
                output_h, output_w = output.size(-2), output.size(-1)
                mask_h, mask_w = data.size(-2), data.size(-1)
                if output_h != mask_h or output_w != mask_w:
                    output = F.interpolate(output, size=(mask_h, mask_w), mode="bilinear")

                if self.args.multi_task:
                    total_loss, loss_dict, var_dict = self.criterion[0](output, target)
                    for loss_fn in loss_dict.keys():  # [bce_with_logit, ...]
                        logging_name = loss_fn
                        # print(self.train_metrics.result())
                        # print(self.log_metrics.result())
                        self.train_metrics.update(logging_name, loss_dict[logging_name].item())  # metric_fn마다 값 update
                        self.log_metrics.update(f"{logging_name}_var", var_dict[f"{logging_name}_var"].item())
                else:
                    for loss_fn in self.criterion:  # [bce_with_logit, ...]
                        loss = loss_fn(output, target)
                        # print(f"{loss_fn} : ", loss)
                        self.train_metrics.update(loss_fn.__class__.__name__, loss.item())  # metric_fn마다 값 update
                        total_loss += loss

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

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
            if self.args.multi_task:
                log_var_dict = self.log_metrics.result()
                wandb.log({"Epoch_" + k: v for k, v in log_var_dict.items()})

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
            if self.args.lr_scheduler["type"] == "ReduceLROnPlateau":
                self.lr_scheduler.step(log["val_Loss"])
            else:
                self.lr_scheduler.step()

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

                if self.args.multi_task:
                    total_val_loss, _, _ = self.criterion[0](output, target)
                else:
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
