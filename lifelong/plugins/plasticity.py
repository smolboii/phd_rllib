from operator import attrgetter
from typing import List, Union, Callable
from copy import deepcopy, copy
import os
import logging
import math

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot
import mnist
import cv2

from lifelong.core.lifelong_learner import LifelongLearner
from lifelong.models.wrappers.base import ModelWrapper
from .base import LifelongLearnerPlugin
from lifelong.util import shuffle_tensors

class PlasticityTrackerPlugin(LifelongLearnerPlugin):

    def __init__(self, obs_shape: tuple, log_dir: str, freq: int = 25, train_epochs: int = 100,
                 batch_size: int = 256, logger: logging.Logger = logging.getLogger("")):
        super().__init__(logger=logger)

        self.obs_shape = obs_shape
        self.log_dir = log_dir
        self.freq = freq
        self.train_epochs = train_epochs
        self.batch_size = batch_size

        os.makedirs(log_dir)
        self.tb_writer = SummaryWriter(log_dir)

        self.dataset = mnist.MNIST("./mnist_data", return_type="numpy")
        self.perm_inds = None
        self.train_images = None
        self.train_labels = None
        self.train_targets = None

        self.init_loss = None
        self.init_accuracy = None

    def before_learn(self, lifelong_learner: LifelongLearner):
        super().before_learn(lifelong_learner)
        self._gen_data(lifelong_learner)

    def _process_mnist(self, train: bool):
        images = self.dataset.train_images if train else self.dataset.test_images
        labels = self.dataset.train_labels if train else self.dataset.test_labels

        # permute images to get "permuted MNIST"
        im_len = len(images[0])
        self.perm_inds = self.perm_inds if self.perm_inds != None else torch.randperm(im_len)
        images[:,torch.arange(im_len)] = images[:,self.perm_inds]

        n_ims = len(images)
        proc_images = np.zeros((n_ims, 84, 84, 4))
        for i, im in enumerate(images):
            rsz_im = cv2.resize(np.reshape(im.astype(np.uint8), (28, 28)), (84, 84))
            proc_images[i] = np.stack(([rsz_im]*4), axis=-1)

        #proc_images = torch.from_numpy(proc_images).float()
        proc_images = torch.randint(0, 255, proc_images.shape)
        targets = np.zeros((n_ims, 18))
        targets[np.arange(n_ims), labels] = 1  # index to one-hot vector encoding (the last 8 outputs will always be 0, since there are only 10 digit classes)
        targets = torch.from_numpy(targets).float()

        return proc_images, targets

    def _gen_data(self, lifelong_learner: LifelongLearner):
    
        self.dataset.load_training()
        self.dataset.load_testing()

        # process data
        # self.train_images, self.train_targets = self._process_mnist(True)
        # self.test_images, self.test_targets = self._process_mnist(False)
        # self.test_labels = torch.from_numpy(self.dataset.test_labels)

        n_ims = 100_000
        self.train_images = torch.randn((n_ims, 84, 84, 4))
        self.train_targets = torch.zeros((n_ims, 18))
        self.train_labels = torch.randint(0, 18, (n_ims,))
        self.train_targets[torch.arange(n_ims), self.train_labels] = 1

    def _eval(self, model_wrapper: ModelWrapper, device: str):

        gpu_test_images = self.test_images.to(device)
        gpu_test_labels = self.test_labels.to(device)

        bs = 1024
        preds = torch.zeros(self.test_targets.shape, device=device)
        for i in range(math.ceil(len(self.test_images) / bs)):
            slice_inds = slice(i*bs, (i+1)*bs)
            preds[slice_inds] = model_wrapper.forward(gpu_test_images[slice_inds])

        pred_labels = torch.argmax(preds, dim=1).to(device)
        n_correct = (pred_labels == gpu_test_labels).int().sum().cpu().item()
        accuracy = n_correct / len(self.test_images)

        return accuracy

    def during_knowledge_distillation(self, kd_epoch: int, env_name: str, lifelong_learner: LifelongLearner):
        super().during_knowledge_distillation(kd_epoch, env_name, lifelong_learner)

        if lifelong_learner.global_sleep_epoch % self.freq != 0:
            return
        
        model_wrapper = lifelong_learner.sleep_algo_callback_wrapper.instantiate_model(lifelong_learner.env_names[0], lifelong_learner.device)
        model_wrapper.model.load_state_dict(lifelong_learner.sleep_model_wrapper.model.state_dict())

        optim = torch.optim.Adam(model_wrapper.model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()

        amp_enabled = True
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        final_loss = 0
        final_accuracy = 0
        n_train = len(self.train_images)
        for epoch in range(self.train_epochs):
            gpu_train_images, gpu_train_targets, gpu_train_labels = (t.to(lifelong_learner.device) for t in 
                                                                     shuffle_tensors(self.train_images, self.train_targets, self.train_labels))

            total_loss = 0
            total_correct = 0
            bs = self.batch_size
            for i in range(len(gpu_train_images) // bs):

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    slice_inds = slice(i*bs, (i+1)*bs)
                    batch = gpu_train_images[slice_inds]
                    preds = model_wrapper.forward(batch)
                    targets = gpu_train_targets[slice_inds]
                    loss = loss_fn.forward(preds, targets)

                optim.zero_grad()
                loss.backward()
                optim.step()

                loss = loss.cpu().detach().item()
                total_loss += loss

                pred_labels = torch.argmax(preds, dim=1)
                total_correct += torch.sum(pred_labels==gpu_train_labels[slice_inds]).int().cpu().detach().item()

            total_loss /= (i+1)

            tag_name = f"ep_{lifelong_learner.global_sleep_epoch}"
            accuracy = total_correct / n_train
            self.tb_writer.add_scalars("accuracy_curves/all", {tag_name: accuracy}, global_step=epoch)
            self.tb_writer.add_scalar(f"accuracy_curves/{tag_name}", accuracy, global_step=epoch)

            self.tb_writer.add_scalars("loss_curves/all", {tag_name: total_loss}, global_step=epoch)
            self.tb_writer.add_scalar(f"loss_curves/{tag_name}", total_loss, global_step=epoch)

            final_loss = total_loss
            final_accuracy = accuracy

        self.tb_writer.add_scalar("final_loss", final_loss, global_step=lifelong_learner.global_sleep_epoch)
        self.tb_writer.add_scalar("final_accuracy", final_accuracy, global_step=lifelong_learner.global_sleep_epoch)

        if self.init_loss is None:
            self.init_loss = final_loss
            self.init_accuracy = final_accuracy
        
        self.tb_writer.add_scalar("plasticity_ratio", round(final_accuracy / self.init_accuracy, 5), global_step=lifelong_learner.global_sleep_epoch)
                


        

        