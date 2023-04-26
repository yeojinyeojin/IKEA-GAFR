import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.chamfer_distance import ChamferDistance
from pytorch3d.ops import knn_points
from collections import defaultdict

class Metric(nn.Module):
    def __init__(self, class_dict, prefix):
        super().__init__()
        self.class_dict =  class_dict
        self.chamfer_dist = ChamferDistance()
        self.prefix = prefix
        self.seg_criterion = torch.nn.CrossEntropyLoss()

        assert self.prefix in ['train', 'val', 'test']

        self.reset_state()

    def evaluate_chamfer_and_f1(self, pred_pc, gt_pc, metadata, m_name=None):
        def evaluate_f1(dis_to_pred, dis_to_gt, pred_length, gt_length, thresh):
            recall = np.sum(dis_to_gt < thresh) / gt_length
            prec = np.sum(dis_to_pred < thresh) / pred_length
            return 2 * prec * recall / (prec + recall + 1e-8)

        if m_name is None:
            return

        batchsize = len(gt_pc)
        for i in range(batchsize):
            idx, cls_labels = metadata[i]
            cls = f"{idx}_{m_name}_{i}"  # metas[i].split('/')[0]
            
            self.model_number_dict[cls] += 1
            d1, d2, i1, i2 = self.chamfer_dist(pred_pc[i].unsqueeze(0), gt_pc[i].unsqueeze(0))
            d1, d2 = d1.cpu().numpy(), d2.cpu().numpy()
            self.cd_dict[cls] += np.mean(d1) + np.mean(d2)
            self.f1_tau_dict[cls] += evaluate_f1(d1, d2, pred_pc[i].size(0), gt_pc[i].size(0), 1e-4)
            self.f1_2tau_dict[cls] += evaluate_f1(d1, d2, pred_pc[i].size(0), gt_pc[i].size(0), 2e-4)

    def loss_points(self, pred_pointclouds, gt_pointclouds, weighted_sample=True, percent=0.5, weight=3.0):
        batchsize = len(gt_pointclouds)
        cd_list = []
        for i in range(batchsize):
            pred_pointcloud = pred_pointclouds[i]
            gt_pointcloud = gt_pointclouds[i]
            dist1, dist2, _, _ = self.chamfer_dist(pred_pointcloud[None, ...], gt_pointcloud[None, ...])
            dist1 = torch.sqrt(dist1)
            dist2 = torch.sqrt(dist2)
            loss_cd = (torch.mean(dist1)) + (torch.mean(dist2))
            if weighted_sample:
                dist1_weighted, dist1_weighted_idx = torch.topk(dist1, int(percent * pred_pointcloud.shape[0]),
                                                                largest=True)
                dist2_weighted, dist2_weighted_idx = torch.topk(dist2, int(percent * gt_pointcloud.shape[0]),
                                                                largest=True)
                loss_weighted = (torch.mean(dist1[:, dist1_weighted_idx[0]])) + (
                    torch.mean(dist2[:, dist2_weighted_idx[0]]))
                loss_cd += weight * loss_weighted
            cd_list.append(loss_cd)
        return sum(cd_list) / len(cd_list)

    def loss_density(self, pred_density, gt_density):
        return F.l1_loss(pred_density, gt_density)

    def loss_segmented_density(self, pred_seg_density, gt_seg_density):
        # return F.hinge_embedding_loss(input=pred_seg_density, target=gt_seg_density)
        # return F.kl_div(input=pred_seg_density, target=gt_seg_density)
        return F.binary_cross_entropy(input=pred_seg_density.to(torch.float), target=gt_seg_density.to(torch.float))

    def loss_segmentation(self, pred_pts, gt_pts, pred_cls, metadatas):
        # https://arxiv.org/pdf/1810.00461.pdf - page 6 for ref
        batchsize = len(gt_pts)
        seg_list = []
        for pred_pt, gt_pt, pred_cl, (_, gt_cl) in zip(pred_pts, gt_pts, pred_cls, metadatas):
            fwd_idx = knn_points(gt_pt.unsqueeze(0), pred_pt.unsqueeze(0), K=1)[1].squeeze()
            bwd_idx = knn_points(pred_pt.unsqueeze(0), gt_pt.unsqueeze(0), K=1)[1].squeeze()

            fwd_tgt = torch.tensor(gt_cl, dtype=torch.long, device=pred_cl.device).squeeze()
            bwd_tgt = torch.tensor(gt_cl, dtype=torch.long, device=pred_cl.device)[bwd_idx].squeeze()
            
            fwd_loss = self.seg_criterion(input=pred_cl[fwd_idx], target=fwd_tgt)
            bwd_loss = self.seg_criterion(input=pred_cl, target=bwd_tgt)

            seg_loss = (fwd_loss + bwd_loss) / 2
            seg_list.append(seg_loss)
        
        return sum(seg_list) / len(seg_list)

    def reset_state(self):
        self.model_number_dict = defaultdict(float, {i: 0 for i in self.class_dict})
        self.cd_dict = defaultdict(float, {i: 0 for i in self.class_dict})
        self.f1_tau_dict = defaultdict(float, {i: 0 for i in self.class_dict})
        self.f1_2tau_dict = defaultdict(float, {i: 0 for i in self.class_dict})

    def get_dict(self):
        res_dict = {}
        mean_loss = []
        for item in self.model_number_dict:
            number = self.model_number_dict[item] + 1e-8
            try:
                if self.cd_dict[item] != 0:
                    res_dict['{}_{}_cd'.format(self.prefix, self.class_dict[item])] = (self.cd_dict[item] / number)
                    res_dict['{}_{}_f1_tau'.format(self.prefix, self.class_dict[item])] = (self.f1_tau_dict[item] / number)
                    res_dict['{}_{}_f1_2tau'.format(self.prefix, self.class_dict[item])] = (self.f1_2tau_dict[item] / number)
                    mean_loss.append(self.cd_dict[item] / number)
            except Exception as err:
                res_dict[f'{item}_cd'] = (self.cd_dict[item] / number)
                res_dict[f'{item}_f1_tau'] = (self.f1_tau_dict[item] / number)
                res_dict[f'{item}_f1_2tau'] = (self.f1_2tau_dict[item] / number)
                mean_loss.append(self.cd_dict[item] / number)
                
        mean_loss = np.mean(mean_loss)
        res_dict["{}_loss".format(self.prefix)] = mean_loss
        return res_dict
