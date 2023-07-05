import torch
import torch.nn as nn


def gram_matrix(feature_maps):
    """
    feature_maps: b, c, h, w
    gram_matrix: b, c, c
    """
    b, c, h, w = feature_maps.size()
    features = feature_maps.view(b, c, h * w)
    G = torch.bmm(features, torch.transpose(features, 1, 2))
    G.div_(h*w) # 输入是[0-255]时可以加
    return G


def gram_loss(content_features, target_features, weights=[1 for n in [64,128,256,512,512]]):
    gram_loss = 0.0
    for i, (c_feats, s_feats) in enumerate(zip(content_features, target_features)):
        gram_loss += weights[i] * torch.mean(torch.abs(gram_matrix(c_feats) - gram_matrix(s_feats)))
    return gram_loss


def tv_loss(rgb: torch.Tensor):
    w_variance = torch.mean(torch.pow(rgb[:, :, :, :-1] - rgb[:, :, :, 1:], 2))
    h_variance = torch.mean(torch.pow(rgb[:, :, :-1, :] - rgb[:, :, 1:, :], 2))
    return (h_variance + w_variance) / 2.0


if __name__ == '__main__':
    x = torch.linspace(0, 1, 64*3).reshape(3, 8, 8)
    x = x[None,...] 

    print(gram_matrix(x))



