import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.ops import (
    GraphConv,
    SubdivideMeshes,
    vert_align,
)


class MeshRefinementStage(nn.Module):
    def __init__(self, img_feat_dim, vert_feat_dim, hidden_dim, stage_depth, gconv_init="normal"):
        """
        Args:
          img_feat_dim: Dimension of features we will get from vert_align
          vert_feat_dim: Dimension of vert_feats we will receive from the
                        previous stage; can be 0
          hidden_dim: Output dimension for graph-conv layers
          stage_depth: Number of graph-conv layers to use
          gconv_init: How to initialize graph-conv layers
        """
        super(MeshRefinementStage, self).__init__()

        # fc layer to reduce feature dimension
        self.bottleneck = nn.Linear(img_feat_dim, hidden_dim)

        # deform layer
        self.verts_offset = nn.Linear(hidden_dim + 3, 3)

        # graph convs
        self.gconvs = nn.ModuleList()
        for i in range(stage_depth):
            if i == 0:
                input_dim = hidden_dim + vert_feat_dim + 3
            else:
                input_dim = hidden_dim + 3
            gconv = GraphConv(input_dim, hidden_dim, init=gconv_init, directed=False)
            self.gconvs.append(gconv)

        # initialization
        nn.init.normal_(self.bottleneck.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bottleneck.bias, 0)

        nn.init.zeros_(self.verts_offset.weight)
        nn.init.constant_(self.verts_offset.bias, 0)

    def forward(self, x, mesh, vert_feats=None):
        img_feats = vert_align(x, mesh, return_packed=True, padding_mode="border")
        # 256 -> hidden_dim
        img_feats = F.relu(self.bottleneck(img_feats))
        if vert_feats is None:
            # hidden_dim + 3
            vert_feats = torch.cat((img_feats, mesh.verts_packed()), dim=1)
        else:
            # hidden_dim * 2 + 3
            vert_feats = torch.cat((vert_feats, img_feats, mesh.verts_packed()), dim=1)

        vert_feats_nopos = None
        for graph_conv in self.gconvs:
            vert_feats_nopos = F.relu(graph_conv(vert_feats, mesh.edges_packed()))
            vert_feats = torch.cat((vert_feats_nopos, mesh.verts_packed()), dim=1)

        # refine
        deform = torch.tanh(self.verts_offset(vert_feats))
        mesh = mesh.offset_verts(deform)
        return mesh, vert_feats_nopos


class MeshRCNNGraphConvHead(nn.Module):
    """
    A mesh head with vert align, graph conv layers and refine layers.
    """

    def __init__(self, cfg, input_shape):
        super(MeshRCNNGraphConvHead, self).__init__()

        # fmt: off
        num_stages         = cfg.MODEL.ROI_MESH_HEAD.NUM_STAGES
        num_graph_convs    = cfg.MODEL.ROI_MESH_HEAD.NUM_GRAPH_CONVS  # per stage
        graph_conv_dim     = cfg.MODEL.ROI_MESH_HEAD.GRAPH_CONV_DIM
        graph_conv_init    = cfg.MODEL.ROI_MESH_HEAD.GRAPH_CONV_INIT
        input_channels     = input_shape.channels
        # fmt: on

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            vert_feat_dim = 0 if i == 0 else graph_conv_dim
            stage = MeshRefinementStage(
                input_channels,
                vert_feat_dim,
                graph_conv_dim,
                num_graph_convs,
                gconv_init=graph_conv_init,
            )
            self.stages.append(stage)

    def forward(self, x, mesh):
        if x.numel() == 0 or mesh.isempty():
            return [Meshes(verts=[], faces=[])]

        meshes = []
        vert_feats = None
        for stage in self.stages:
            mesh, vert_feats = stage(x, mesh, vert_feats=vert_feats)
            meshes.append(mesh)
        return meshes