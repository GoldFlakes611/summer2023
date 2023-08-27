import numpy as np
import torch

def direction_metric(pred, act):
    angle_true = act[:, 0]
    angle_pred = pred[:, 0]
    turns = torch.abs(angle_true) > 0.1
    logits = torch.sign(angle_pred[turns]) == torch.sign(angle_true[turns])
    return torch.sum(logits.float()), len(logits)

def angle_metric(pred, act):
    angle_true = act[:, 0]
    angle_pred = pred[:, 0]
    logits = torch.abs(angle_true - angle_pred) < 0.1
    return torch.mean(logits.float())

def loss_fn(steering, throttle, steering_pred, throttle_pred, throttle_weight):
    steering_loss = ((steering - steering_pred)**2).mean()
    throttle_loss = ((throttle - throttle_pred)**2).mean()
    loss = steering_loss + throttle_weight * throttle_loss
    return loss