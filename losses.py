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
    G.div_(h*w)
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


###############################################################################
###############################arf utils#######################################
###############################################################################
def cos_distance(a, b, center=True):
    """a: [b, c, hw],
    b: [b, c, h2w2]
    """
    # """cosine distance
    if center:
        a = a - a.mean(2, keepdims=True)
        b = b - b.mean(2, keepdims=True)

    a_norm = ((a * a).sum(1, keepdims=True) + 1e-8).sqrt()
    b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()

    a = a / (a_norm + 1e-8)
    b = b / (b_norm + 1e-8)

    d_mat = 1.0 - torch.matmul(a.transpose(2, 1), b)
    # """"

    """
    a_norm_sq = (a * a).sum(1).unsqueeze(2)
    b_norm_sq = (b * b).sum(1).unsqueeze(1)

    d_mat = a_norm_sq + b_norm_sq - 2.0 * torch.matmul(a.transpose(2, 1), b)
    """
    return d_mat


def cos_loss(a, b):
    # """cosine loss
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()
    # """

    # return ((a - b) ** 2).mean()


def feat_replace(a, b):
    n, c, h, w = a.size()
    n2, c, h2, w2 = b.size()

    assert (n == 1) and (n2 == 1)

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    b_ref = b_flat.clone()

    z_new = []

    # Loop is slow but distance matrix requires a lot of memory
    for i in range(n):
        z_dist = cos_distance(a_flat[i : i + 1], b_flat[i : i + 1])

        z_best = torch.argmin(z_dist, 2)
        del z_dist

        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        feat = torch.gather(b_ref, 2, z_best)

        z_new.append(feat)

    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    return z_new


def guided_feat_replace(a, b, trgt):
    n, c, h, w = a.size()
    n2, c2, h2, w2 = b.size()
    n3, c3, h3, w3 = trgt.size()
    assert (n == 1) and (n2 == 1) and (c == c2) and (n3 == 1) and (h2 == h3) and (w2 == w3)

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c2, -1)
    trgt = trgt.view(n3, c3, -1)

    z_new = []

    # Loop is slow but distance matrix requires a lot of memory
    for i in range(n):
        z_dist = cos_distance(a_flat[i : i + 1], b_flat[i : i + 1])

        z_best = torch.argmin(z_dist, 2)
        del z_dist

        z_best = z_best.unsqueeze(1).repeat(1, c3, 1)
        feat = torch.gather(trgt, 2, z_best)

        z_new.append(feat)

    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c3, h, w)
    return z_new
###############################################################################
###############################arf utils#######################################
###############################################################################


if __name__ == '__main__':
    x = torch.linspace(0, 1, 64*3).reshape(3, 8, 8)
    x = x[None,...] 

    print(gram_matrix(x))



