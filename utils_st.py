import torch

## EFDM
def exact_feature_distribution_matching(content_feat, style_feat):
    assert (content_feat.size() == style_feat.size())
    B, C, W, H = content_feat.size(0), content_feat.size(1), content_feat.size(2), content_feat.size(3)
    value_content, index_content = torch.sort(content_feat.view(B,C,-1))  # sort conduct a deep copy here.
    value_style, _ = torch.sort(style_feat.view(B,C,-1))  # sort conduct a deep copy here.
    inverse_index = index_content.argsort(-1)
    new_content = content_feat.view(B,C,-1) + (value_style.gather(-1, inverse_index) - content_feat.view(B,C,-1).detach())
    return new_content.view(B, C, W, H)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std