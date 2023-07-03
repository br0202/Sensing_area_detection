import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class TwinLoss(nn.modules.Module):
    def __init__(self, L1_w=0.5, L2_w=0.5, L_left_w=0.5, L_right_w=0.5):
        super(TwinLoss, self).__init__()
        self.L1_w = L1_w
        self.L2_w = L2_w
        self.L_left_w = L_left_w
        self.L_right_w = L_right_w
        self.loss = nn.MSELoss()

    def BCE_Loss(self, pred, gt):
        BCE_loss = nn.BCELoss()
        loss = BCE_loss(pred, gt)
        return loss

    def forward(self, left_pred, left_gt):
        """
        Args:xx
        Return:
            (float): The loss
        """

        # left_loss = self.L1_w * self.L1_loss(left_pred, left_gt) + self.L2_w * self.L2_loss(right_pred, right_gt)
        # right_loss = self.L1_w * self.L1_loss(right_pred, right_gt) + self.L2_w * self.L2_loss(right_pred, right_gt)
        # loss = self.L_right_w * left_loss + self.L_right_w * right_loss

        # loss = self.L1_w * self.BCE_Loss(left_pred, left_gt) + self.L2_w * self.BCE_Loss(right_pred, right_gt)
        loss = self.loss(left_pred, left_gt)

        return loss

