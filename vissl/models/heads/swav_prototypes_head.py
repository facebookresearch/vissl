# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn


class SwAVPrototypesHead(nn.Module):
    """
    SwAV head used in https://arxiv.org/pdf/2006.09882.pdf paper.

    The head is composed of 2 parts
        1) projection of features to lower dimension like 128
        2) feature classification into clusters (also called prototypes)

    The projected features are L2 normalized before clustering step.

    Input: 4D torch.tensor of shape (N x C x H x W)

    Output: List(2D torch.tensor of shape N x num_clusters)
    """

    def __init__(self, model_config, dims, use_bn, num_clusters, use_bias=True):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
            dims (int): dimensions of the linear layer. Must have length at least 2.
                        Example: [2048, 2048, 128] attaches linear layer
                                 Linear(2048, 2048) -> BN -> Relu -> Linear(2048, 128)
            use_bn (bool): whether to attach BatchNorm after Linear layer
            num_clusters (List(int)): number of prototypes or clusters. Typically 3000.
                                      Example dims=[3000] will attach 1 prototype head.
                                              dims=[3000, 3000] will attach 2 prototype heads
            use_bias (bool): whether the Linear layer should have bias or not
        """

        super().__init__()
        # build the projection head
        layers = []
        last_dim = dims[0]
        for i, dim in enumerate(dims[1:]):
            layers.append(nn.Linear(last_dim, dim, bias=use_bias))
            if i == len(dims) - 2:
                break
            if use_bn:
                layers.append(
                    nn.BatchNorm1d(
                        dim,
                        eps=model_config.HEAD.BATCHNORM_EPS,
                        momentum=model_config.HEAD.BATCHNORM_MOMENTUM,
                    )
                )
            layers.append(nn.ReLU(inplace=True))
            last_dim = dim
        self.projection_head = nn.Sequential(*layers)

        # prototypes (i.e. centroids) layers
        if len(num_clusters) > 0:
            self.nmb_heads = len(num_clusters)
            for i, k in enumerate(num_clusters):
                self.add_module(
                    "prototypes" + str(i), nn.Linear(dims[-1], k, bias=False)
                )
        else:
            self.nmb_heads = 0

    def forward(self, batch):
        """
        Args:
            batch (4D torch.tensor): shape (N x C x H x W)
        Returns:
            List(2D torch.tensor of shape N x num_clusters)
        """
        batch = self.projection_head(batch)

        batch = nn.functional.normalize(batch, dim=1, p=2)

        out = [batch]
        if self.nmb_heads > 0:
            for i in range(self.nmb_heads):
                out.append(getattr(self, "prototypes" + str(i))(batch))

        return out
