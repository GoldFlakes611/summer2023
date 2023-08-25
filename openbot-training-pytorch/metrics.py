'''
Name: metrics.py
Description: Metrics for evaluating model performance
Date: 2023-08-25
Last Modified: 2023-08-25
'''
import torch

def direction_metric(pred : torch.Tensor, act : torch.Tensor) -> torch.Tensor:
    '''
    Direction metric - how well the model predicts the direction of the car
    Args:
        pred (torch.Tensor): predicted values
        act (torch.Tensor): actual values
    Returns:
        torch.Tensor: direction metric
    '''
    angle_true = act[:, 0]
    angle_pred = pred[:, 0]
    turns = torch.abs(angle_true) > 0.1
    logits = torch.sign(angle_pred[turns]) == torch.sign(angle_true[turns])
    return torch.sum(logits.float()), len(logits)

def angle_metric(pred : torch.Tensor, act : torch.Tensor) -> torch.Tensor:
    '''
    Angle metric - how well the model predicts the angle of the car
    Args:
        pred (torch.Tensor): predicted values
        act (torch.Tensor): actual values
    Returns:
        torch.Tensor: angle metric
    '''
    angle_true = act[:, 0]
    angle_pred = pred[:, 0]
    logits = torch.abs(angle_true - angle_pred) < 0.1
    return torch.mean(logits.float())

def loss_fn(steering : torch.Tensor, throttle : torch.Tensor, steering_pred : torch.Tensor, throttle_pred : torch.Tensor, throttle_weight : torch.Tensor) -> torch.Tensor:
    '''
    Loss function for the model
    Args:
        steering (torch.Tensor): steering values
        throttle (torch.Tensor): throttle values
        steering_pred (torch.Tensor): predicted steering values
        throttle_pred (torch.Tensor): predicted throttle values
        throttle_weight (float): weight for throttle
    Returns:
        torch.Tensor: loss
    '''
    steering_loss = ((steering - steering_pred)**2).mean()
    throttle_loss = ((throttle - throttle_pred)**2).mean()
    loss = steering_loss + throttle_weight * throttle_loss
    return loss
