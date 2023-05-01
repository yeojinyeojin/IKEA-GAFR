from typing import Any, List, Sequence, Tuple, Union

import os
import cv2
import pickle as pkl
import numpy as np
import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.metric import Metric
from common.networks import define_G, get_norm_layer, init_net, UnitBlock, ResMLP
from common.schedulers import get_decay_scheduler
from common.utils import get_classdict
from omegaconf import DictConfig
from torch.optim import Optimizer


class SketchTranslator(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = define_G(input_nc=1, output_nc=64 + 1, ngf=64, netG='resnet_9blocks', norm='instance',
                                  use_dropout=False, init_type='normal', init_gain=0.02)

    def forward(self, sketch):
        feature1 = self.parameter.model[:22](sketch)
        feature2 = self.parameter.model[22:25](feature1)
        feature3 = self.parameter.model[25:28](feature2)
        feature4 = self.parameter.model[28:](feature3)
        return [feature1, feature2, feature3, feature4]

"""
class DensityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = UnitBlock(512 + 256 + 128 + 64, 256)
        self.conv = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.block2 = UnitBlock(64, 1)

    def forward(self, features):
        _, _, h_out, w_out = features[-1].shape
        for i in range(len(features) - 1):
            features[i] = F.interpolate(features[i], (h_out, w_out))
        input_feature = torch.cat(features, dim=1)
        m = self.block1(input_feature)
        m = self.relu(self.conv(m))
        m = self.block2(m)
        m = m / torch.sum(m, dim=(2, 3), keepdim=True)
        return m


class MapSampler(nn.Module):
    def __init__(self, mode='stable'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['stable', 'normal']

    def forward(self, density_map, pt_count):
        if self.mode == 'stable':
            return self.sample_stable(density_map, pt_count)
        elif self.mode == 'normal':
            return self.sample_normal(density_map, pt_count)
        else:
            raise NotImplementedError

    def sample_stable(self, density_map, pt_count):
        density_map = density_map * pt_count
        H, W = density_map.shape[-2:]
        density_map = torch.round(density_map).long()
        points = [torch.where(img[0] >= 1 - 1e-3) for img in density_map]
        density = [b_map[0][b_point] for b_map, b_point in zip(density_map, points)]
        aux_norm = torch.tensor([(W - 1), (H - 1)], device=density_map.device).view(1, 2)
        pointclouds_normed = [torch.stack(point, dim=-1).flip(-1).float() * 2 / aux_norm - 1
                              for point in points]
        xys = [torch.repeat_interleave(pointcloud_normed, b_density, dim=0) for pointcloud_normed, b_density in
               zip(pointclouds_normed, density)]
        random_prior = [torch.rand(size=(xy.shape[0], 1), device=density_map.device) for xy
                        in
                        xys]
        return xys, random_prior

    def sample_normal(self, density_map, pt_count):
        H, W = density_map.shape[-2:]
        indices = torch.multinomial(density_map[:, 0, :, :].flatten(1), pt_count, replacement=True)
        xs = indices % W
        ys = indices // W
        xys = torch.stack((xs, ys), -1)
        aux_norm = torch.tensor([(W - 1), (H - 1)], device=density_map.device).view(1, 2)
        xys = xys.float() * 2 / aux_norm - 1
        random_prior = torch.rand(size=(xys.shape[0], xys.shape[1], 1), device=density_map.device)
        return xys, random_prior


class DepthHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_encode1 = ResMLP(1, 16, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())
        self.z_encode2 = ResMLP(16, 32, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())
        self.z_encode3 = ResMLP(32, 64, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())

        self.z_decode1 = ResMLP(512 + 256 + 128 + 64 + 64, 64, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())
        self.z_decode2 = ResMLP(64, 32, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())
        self.z_decode3 = ResMLP(32, 16, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())
        self.gen_z = nn.Linear(16, 1)

    def unproj(self, WH, Z):
        w, h = WH.T
        z = Z.squeeze(1)
        Y = -h * 0.75
        X = w * 0.75
        res = torch.stack([X, Y, z], dim=1)
        return res

    def forward(self, features, WH, random_prior):
        batchsize = features[0].shape[0]
        pointclouds = []
        for i in range(batchsize):
            feats_i = [feat[i].unsqueeze(0) for feat in features]
            wh_i = WH[i].unsqueeze(0)
            random_prior_i = random_prior[i].unsqueeze(0)
            z_i = self.z_encode1(random_prior_i)
            z_i = self.z_encode2(z_i)
            z_i = self.z_encode3(z_i)

            feats_sampled_i = [F.grid_sample(
                feat_i, wh_i.unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                                   .permute(0, 2, 1) for feat_i in feats_i]
            feats_i = torch.cat(feats_sampled_i, dim=-1)
            z_i = self.z_decode1(torch.cat([feats_i, z_i], dim=-1))
            z_i = self.z_decode2(z_i)
            z_i = self.z_decode3(z_i)
            z_i = self.gen_z(z_i)
            z_i -= z_i.mean(dim=1, keepdim=True)
            pointclouds.append(self.unproj(wh_i.squeeze(0), z_i.squeeze(0)))
        return pointclouds
"""

class SketchSampler(pl.LightningModule):
    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.save_hyperparameters()

        # hparams
        self.log_metric = True
        self.seg_classes = 24
        self.emb_dim = 16
        self.include_rand_prior = True
        self.use_seg_density_loss = True
        
        self.depth_head = init_net(IkeaDepthHead(include_prior=self.include_rand_prior, 
                                                 emb_dim=self.emb_dim,
                                                 seg_classes=self.seg_classes))
        self.density_head = init_net(IkeaDensityHead(seg_classes=self.seg_classes))
        self.sketch_translator = SketchTranslator()
        self.map_sampler = IkeaMapSampler(include_prior=self.include_rand_prior, 
                                          seg_classes=self.seg_classes, 
                                          emb_dim=self.emb_dim)

        self.n_points = self.cfg.train.n_points
        self.lambda1 = self.cfg.train.lambda1
        self.lambda2 = self.cfg.train.lambda2
        self.lambda3 = self.cfg.train.lambda3
        self.lambda_seg = self.cfg.train.lambda_seg

        self.class_dict = get_classdict()
        self.train_metric = Metric(self.class_dict, 'train')
        self.val_metric = Metric(self.class_dict, 'val')
        self.test_metric = Metric(self.class_dict, 'test')

    def forward(self, sketch, density_map, use_predicted_map):
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        """
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data.shape)

        features = self.sketch_translator(sketch)
        predicted_map = self.density_head(features)

        if use_predicted_map or density_map is None:
            WH, depth_prior = self.map_sampler(predicted_map, self.n_points)
        else:
            WH, depth_prior = self.map_sampler(density_map, self.n_points)
        predicted_points, predicted_clss = self.depth_head(features, WH, depth_prior)
        return predicted_map, predicted_points, predicted_clss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        sketch, pointclouds, density_map, metadata = batch
        predicted_map, predicted_points, predicted_clss = self(sketch, density_map, use_predicted_map=False)
        
        points_loss = self.train_metric.loss_points(predicted_points, pointclouds)
        seg_loss = self.train_metric.loss_segmentation(predicted_points, pointclouds, predicted_clss, metadata)
        # completeness_all = self.train_metric.eval_completeness(predicted_points, pointclouds)
        # accuracy_all = self.train_metric.eval_accuracy(predicted_points, pointclouds)
        # miou_all = self.train_metric.eval_iou(predicted_points, predicted_clss, pointclouds, metadata)

        if self.use_seg_density_loss:
            density_loss = self.train_metric.loss_density(predicted_map[:, 0:1], density_map[:, 0:1])
            seg_density_loss = self.train_metric.loss_segmented_density(predicted_map[:, 1:], density_map[:, 1:])
            train_loss = self.lambda1 * points_loss + \
                            self.lambda2 * density_loss + \
                            self.lambda_seg * seg_density_loss + \
                            self.lambda3 * seg_loss
        else:
            density_loss = self.train_metric.loss_density(predicted_map, density_map)
            train_loss = self.lambda1 * points_loss + \
                            self.lambda2 * density_loss + \
                            self.lambda3 * seg_loss
        
        # print('******', points_loss, density_loss, seg_loss, train_loss)
        self.log_dict(
            {"train_loss": train_loss,
             "density_loss": density_loss,
             "points_loss": points_loss,
             "seg_loss": seg_loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )
        return train_loss

    def validation_step(self, batch: Any, batch_idx: int):
        sketch, pointclouds, density_map, metadata = batch
        predicted_map, predicted_points, predicted_clss = self(sketch, density_map, use_predicted_map=True)
        
        if self.log_metric:
            self.val_metric.evaluate_chamfer_and_f1(predicted_points, pointclouds, metadata, m_name=f"model_{batch_idx}")
            completeness_all = self.val_metric.eval_completeness(predicted_points, pointclouds)
            accuracy_all = self.val_metric.eval_accuracy(predicted_points, pointclouds)
            miou_all = self.val_metric.eval_iou(predicted_points, predicted_clss, pointclouds, metadata)

        return

    def test_step(self, batch: Any, batch_idx: int):
        sketch, pointclouds, density_map, metadata = batch
        predicted_map, predicted_points, predicted_clss = self(sketch, density_map, use_predicted_map=True)
        
        if self.log_metric:
            self.test_metric.evaluate_chamfer_and_f1(predicted_points, pointclouds, metadata, m_name=f"model_{batch_idx}")
            completeness_all = self.test_metric.eval_completeness(predicted_points, pointclouds)
            accuracy_all = self.test_metric.eval_accuracy(predicted_points, pointclouds)
            miou_all = self.test_metric.eval_iou(predicted_points, predicted_clss, pointclouds, metadata)
            
        output_dir = "/home/niviru/Desktop/FinalProject/IKEA/sketchsampler/outputs_trained_30_epochs/"
        sketch_dir = output_dir + "/sketches"
        gt_pcd_dir = output_dir + "/gt_pcds"
        pred_pcd_dir = output_dir + "/predicted_pcds"
        pred_seg_dir = output_dir + '/predicted_seg_labels'
        print("......WRITING MODELS......")
        for i,s in enumerate(sketch):
            cv2.imwrite(os.path.join(sketch_dir, "{}_{}.png".format(batch_idx, i)), np.uint8(s.cpu().numpy().squeeze()*255))
            #cv2.imwrite(os.path.join(sketch_dir, "{}_{}.png".format(batch_idx, i)), np.uint8(s.cpu().numpy().squeeze()))
            np.save(os.path.join(gt_pcd_dir, "{}_{}.npy".format(batch_idx, i)), pointclouds[i].cpu().numpy())
            np.save(os.path.join(pred_pcd_dir, "{}_{}.npy".format(batch_idx, i)), predicted_points[i].cpu().numpy())
            np.save(os.path.join(pred_seg_dir, "{}_{}.npy".format(batch_idx, i)), predicted_clss[i].cpu().numpy())
            # assert 1 == 2
        
        """
        for i in range(len(metadata)):
            np_pred_pc = predicted_points[i].detach().cpu().numpy()
            # np_pred_pc = to_numpy(predicted_points[i])
            meta = metadata[i]
            cls, obj, view_num = meta.split('/')
            targetPath = os.path.join(self.cfg.train.vis_out_path, 'point_cloud', cls, obj)
            if not os.path.exists(targetPath):
                os.makedirs(targetPath)
            save_path = os.path.join(targetPath, view_num + ".obj")
            pkl.dump(np_pred_pc, open(save_path, 'wb'))
            targetPath = os.path.join(self.cfg.train.vis_out_path, 'density_map', cls, obj)
            if not os.path.exists(targetPath):
                os.makedirs(targetPath)
            save_path = os.path.join(targetPath, view_num + ".dat")
            pkl.dump(density_map[i].detach().cpu().numpy(), open(save_path, 'wb'))
            # pkl.dump(to_numpy(density_map[i]), open(save_path, 'wb'))
        """
        return

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(self.val_metric.get_dict())
        self.val_metric.reset_state()

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(self.test_metric.get_dict())
        self.test_metric.reset_state()

    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(
            self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.cfg.optim.optimizer,
            params=self.parameters()
        )

        if self.cfg.optim.use_lr_scheduler:
            scheduler = [{
                'scheduler': get_decay_scheduler(opt, self.num_training_steps(), 0.9),
                'interval': 'step',
            }]
            return [opt], scheduler

        return opt


class IkeaDensityHead(nn.Module):
    def __init__(self, seg_classes):
        super().__init__()
        self.block1 = UnitBlock(512 + 256 + 128 + 64, 256)
        self.conv = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.block2 = UnitBlock(64, seg_classes)

    def forward(self, features):
        _, _, h_out, w_out = features[-1].shape
        for i in range(len(features) - 1):
            features[i] = F.interpolate(features[i], (h_out, w_out))
        input_feature = torch.cat(features, dim=1)
        m = self.block1(input_feature)
        m = self.relu(self.conv(m))
        m = self.block2(m)  # [4, 24, 256, 256]
        m = m + 1e-12
        
        # exp: can predict self.block2 = UnitBlock(64, seg_classes + 1) instead
        total_m = torch.sum(m, dim=1, keepdim=True)
        total_m = total_m / torch.sum(total_m, dim=(2, 3), keepdim=True)  # [4, 1, 256, 256]
        
        # normalized across channels
        channel_m = m / torch.sum(m, dim=(1), keepdim=True)  # [4, 24, 256, 256]

        fused_m = torch.cat((total_m, channel_m), dim=1)  # [4, 25, 256, 256]
        
        if torch.isnan(fused_m).sum() or torch.isinf(fused_m).sum():
            raise ValueError(f"NAN SUM {torch.isnan(fused_m).sum()}, INF SUM {torch.isinf(fused_m).sum()}")
        return fused_m


class IkeaMapSampler(nn.Module):
    def __init__(self, mode='stable', include_prior=False, seg_classes=16, emb_dim=32):
        super().__init__()
        self.mode = mode

        self.include_prior = include_prior
        self.seg_classes = seg_classes
        self.emb_dim = emb_dim
        self.seg_encoding = nn.Embedding(num_embeddings=seg_classes, embedding_dim=emb_dim)
        assert self.mode in ['stable', 'normal']

    def forward(self, density_map, pt_count):
        if self.mode == 'stable':
            return self.sample_stable(density_map, pt_count)
        elif self.mode == 'normal':
            return self.sample_normal(density_map, pt_count)
        else:
            raise NotImplementedError

    def sample_stable(self, fused_density_map, pt_count):
        density_map = fused_density_map[:, 0:1] * pt_count  # [4, 1, 256, 256]
        H, W = density_map.shape[-2:]
        density_map = torch.round(density_map).long()

        points = [torch.where(img[0] >= 1 - 1e-3) for img in density_map]
        density = [b_map[0][b_point] for b_map, b_point in zip(density_map, points)]
        aux_norm = torch.tensor([(W - 1), (H - 1)], device=density_map.device)

        # flip because we need XYs and not YXs and normalize for zero-centering
        pointclouds_normed = [torch.stack(point, dim=-1).flip(-1).float() * 2 / aux_norm - 1
                              for point in points]
        xys = [torch.repeat_interleave(pointcloud_normed, repeats=b_density, dim=0) for pointcloud_normed, b_density in
               zip(pointclouds_normed, density)]

        # todo: more efficient vectorized way for this instead of wrapper subroutine using torch.gather
        seg_priors = [self._get_seg_prior(b_fused_map[1:], torch.stack(b_point, dim=-1), b_density)
                      for b_fused_map, b_point, b_density in zip(fused_density_map, points, density)]
        return xys, seg_priors

    def _get_seg_prior(self, channel_map, points, density):
        device = channel_map.device
        final_emb_dim = self.emb_dim + 1 if self.include_prior else self.emb_dim

        seg_prior = torch.empty(size=(density.sum(), final_emb_dim), device=device)
        _start = 0
        for pt, c_num_pts in zip(points, density):
            channel_probs = channel_map[:, pt[0], pt[1]]
            samples = torch.multinomial(input=channel_probs, num_samples=c_num_pts, replacement=True)
            # samples = torch.tensor([0] * c_num_pts, device=channel_map.device, dtype=torch.long)
            seg_prior[_start: _start + c_num_pts, :self.emb_dim] = self.seg_encoding(samples)
            _start += c_num_pts

        if self.include_prior:
            seg_prior[:, -1] = torch.rand(len(seg_prior), device=device)
        return seg_prior.to(device)

    def sample_normal(self, density_map, pt_count):
        raise NotImplementedError


class IkeaDepthHead(nn.Module):
    def __init__(self, include_prior, emb_dim, seg_classes):
        super().__init__()
        
        init_dim = emb_dim + 1 if include_prior else emb_dim
        
        self.z_encode1 = ResMLP(init_dim, 32, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())
        self.z_encode2 = ResMLP(32, 64, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())

        self.z_decode1 = ResMLP(512 + 256 + 128 + 64 + 64, 64, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())
        self.z_decode2 = ResMLP(64, 32, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())
        self.z_decode3 = ResMLP(32, 16, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())
        
        self.gen_z = nn.Linear(16, 3)
        self.c_decode = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=seg_classes),
        )
        
    def unproj(self, WH, Z):
        w, h = WH.T
        z = Z.squeeze(1)
        Y = -h * 0.75
        X = w * 0.75
        res = torch.stack([X, Y, z], dim=1)
        return res

    def forward(self, features, WH, random_prior):
        batchsize = features[0].shape[0]
        
        pointcloud_labels = []
        pointclouds = []
        for i in range(batchsize):
            feats_i = [feat[i].unsqueeze(0) for feat in features]
            wh_i = WH[i].unsqueeze(0)
            random_prior_i = random_prior[i].unsqueeze(0)
            z_i = self.z_encode1(random_prior_i)
            z_i = self.z_encode2(z_i)

            feats_sampled_i = [F.grid_sample(
                feat_i, wh_i.unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                                   .permute(0, 2, 1) for feat_i in feats_i]
            feats_i = torch.cat(feats_sampled_i, dim=-1)
            z_i = self.z_decode1(torch.cat([feats_i, z_i], dim=-1))
            z_i = self.z_decode2(z_i)

            z_i_pred = self.z_decode3(z_i)
            z_i_pred = self.gen_z(z_i_pred)
            z_i_pred = z_i_pred - z_i_pred.mean(dim=1, keepdim=True)
            
            c_i = self.c_decode(z_i)
            
            # pointclouds.append(self.unproj(wh_i.squeeze(0), z_i_pred.squeeze(0)))
            pointclouds.append(z_i_pred.squeeze(0))
            pointcloud_labels.append(c_i.squeeze(0))
            
        return pointclouds, pointcloud_labels