import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import binary_dilation
import math
# import torch  # Assuming PyTorch is being used, as in code 2

logger = utils.get_logger()


class Objectview(object):
    def __init__(self, d):
        self.__dict__ = d

    def keys(self):
        return self.__dict__.keys()


def get_norm_flow(lis1, lis2):
    """Get normalized flow"""
    # lis1 and lis2: [B, C=2, 48, 48]

    flow = lis1
    _, _, _h, _w = flow.shape
    flow = torch.cat(
        [flow[:, 0:1] / (_h/2.0), flow[:, 1:2] / (_w/2.0)], 1)
    flow2 = lis2
    _, _, _h, _w = flow2.shape
    flow2 = torch.cat(
        [flow2[:, 0:1] / (_h/2.0), flow2[:, 1:2] / (_w/2.0)], 1)
    return _h, _w, flow, flow2


class FlowAggregationHeadWithResidual(nn.Module):
    """This is based on the original FCN head in the AMD model.
    This module only processes flow info and aggregates info. 

    This has residual in the flow matching.

    Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 args,
                 ssim_sz=1,
                 mask_layer=5,
                 create_flownet=False,
                 flow_feat_before_agg_kernel_size=3,
                 num_flow_feat_channels=64,
                 outlier_robust_loss=False,
                 eps=0.01,
                 q=0.4,
                 mask_size=(48, 48),
                 residual_adjustment_scale=10.,
                 norm_flow=False,
                 clamp_flow_t=None,
                 filter_flow_t=None,
                 free_residual=False,
                 free_residual_with_affine=False,
                 free_residual_with_affine_quadratic=False,
                 object_free_residual=False,
                 free_scale=False,
                 affine_residual=False,
                 allow_residual_resize=False,
                 # Equivalent to making the initialization small (only effective with tanh)
                 pred_div_coeff=10.
                 ):
        self.mask_layer = mask_layer
        super(FlowAggregationHeadWithResidual, self).__init__()
        logger.info("[info] ssim_sz={}".format(ssim_sz))

        self.args = args

        # Although we do not create flow network, we need to assume this to be True to prevent using it to generate segmentation masks.
        assert create_flownet

        self.flow_feat_before_agg = nn.Sequential(
            nn.Conv2d(2, num_flow_feat_channels, kernel_size=flow_feat_before_agg_kernel_size, stride=1,
                      dilation=1,
                      padding=(flow_feat_before_agg_kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_flow_feat_channels, num_flow_feat_channels, kernel_size=flow_feat_before_agg_kernel_size, stride=1,
                      dilation=1,
                      padding=(flow_feat_before_agg_kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.flow_feat_after_agg = nn.Sequential(
            nn.Conv1d(num_flow_feat_channels, num_flow_feat_channels, kernel_size=1, stride=1,
                      dilation=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(num_flow_feat_channels, 2, kernel_size=1, stride=1,
                      dilation=1, bias=True)
        )

        self.num_flow_feat_channels = num_flow_feat_channels

        self.outlier_robust_loss = outlier_robust_loss
        if self.outlier_robust_loss:
            logger.info("Using outlier robust loss")
        self.eps = eps
        self.q = q

        self.mask_size = tuple(mask_size)

        self.residual_adjustment_scale = residual_adjustment_scale

        self.pred_div_coeff = pred_div_coeff

        logger.info(f"Prediction division coefficient: {self.pred_div_coeff}")

        self.norm_flow = norm_flow
        self.clamp_flow_t = clamp_flow_t
        self.filter_flow_t = filter_flow_t

        self.free_residual = free_residual
        self.free_residual_with_affine = free_residual_with_affine
        self.free_residual_with_affine_quadratic = free_residual_with_affine_quadratic
        if self.free_residual_with_affine_quadratic:
            assert self.free_residual_with_affine, "free_residual_with_affine needs to be enabled to enable free_residual_with_affine_quadratic"
        self.object_free_residual = object_free_residual
        self.free_scale = free_scale
        self.affine_residual = affine_residual
        assert (int(self.free_residual) + int(self.free_residual_with_affine) + int(self.object_free_residual) + int(self.free_scale) + int(self.affine_residual)
                ) <= 1, f"Only one of {self.free_residual}, {self.free_residual_with_affine}, {self.object_free_residual}, {self.free_scale}, {self.affine_residual}"
        self.allow_residual_resize = allow_residual_resize

        if self.free_residual_with_affine:
            H, W = mask_size
            coord_map = torch.meshgrid(torch.arange(
                H), torch.arange(W), indexing='ij')
            # Cast to float for ops (e.g. mean)
            if free_residual_with_affine_quadratic:
                num_u_dim = 5
                coord_map = torch.stack((coord_map[0], coord_map[1], coord_map[0] * coord_map[0], coord_map[1]
                                        * coord_map[1], coord_map[0] * coord_map[1]), dim=2).view((-1, num_u_dim)).float().cuda()
            else:
                coord_map = torch.stack(coord_map, dim=2).view(
                    (-1, 2)).float().cuda()

            self.coord_map = coord_map

    def norm_and_clamp_flow(self, flow):
        # flow: [B, C=2, H, W]
        if self.norm_flow:
            flow = flow / flow.abs().max()

        if self.clamp_flow_t is not None:
            flow = flow.clamp(min=-self.clamp_flow_t, max=self.clamp_flow_t)

        if self.filter_flow_t is not None:
            # Remove flow with small values
            flow[flow.abs() < self.filter_flow_t] = 0.

        return flow

    def get_demean_affine_flow(self, mask, flow):
        # mask are expected to sum up to 1 across channel dimension (C=5) before pooling
        B, C, H, W = mask.shape
        # Normalize the mask spatially
        mask_spatial_normalized = mask / mask.sum(dim=(2, 3), keepdim=True)

        # `img_preds` means the mask.
        # Here the scaling of `img_preds_1d` does matter: should sum up to 1 in each channel.
        # img_preds_1d_spatial_normalized: [B, C, H * W]
        img_preds_1d_spatial_normalized = torch.flatten(
            mask_spatial_normalized, 2, 3)

        # flow: [B, 2, H, W]
        # F_u: [B, H * W, 2]
        # ([B, 2, H, W] -> [B, 2, H * W] -> [B, H * W, 2])
        F_u = torch.flatten(flow, 2, 3).permute((0, 2, 1))
        # mu_F: [B, C, 2]
        # mu_F = torch.einsum('b i j, b j k -> b i k', img_preds_1d_spatial_normalized, F_u)
        mu_F = torch.bmm(img_preds_1d_spatial_normalized, F_u)

        # self.coord_map: [H * W, 2]
        # mu_omega: [B, C, 2]
        # TODO: need to check whether these two statements have the same output
        # mu_omega = torch.einsum('b i j, j k -> b i k', img_preds_1d_spatial_normalized, self.coord_map)
        mu_omega = img_preds_1d_spatial_normalized @ self.coord_map

        # F_u_de_mean: [B, C, H * W, 2]
        # ([B, 1, H * W, 2], [B, C, 1, 2] -> [B, C, H * W, 2])
        F_u_de_mean = F_u[:, None, ...] - mu_F[:, :, None, ...]
        # u_de_mean: [B, C, H * W, 2]
        # ([1, 1, H * W, 2], [B, C, 1, 2] -> [B, C, H * W, 2])
        u_de_mean = self.coord_map[None, None, ...] - mu_omega[:, :, None, ...]

        # F_u_demean_u_demean_T: [B, C, H * W, 2, 2]
        F_u_demean_u_demean_T = torch.einsum(
            'b i j k, b i j l -> b i j k l', F_u_de_mean, u_de_mean)
        # sigma_F_omega: [B, C, 2, 2]
        sigma_F_omega = torch.einsum(
            'b i j, b i j k l -> b i k l', img_preds_1d_spatial_normalized, F_u_demean_u_demean_T)

        # u_demean_u_demean_T: [B, C, H * W, 2, 2]
        u_demean_u_demean_T = torch.einsum(
            'b i j k, b i j l -> b i j k l', u_de_mean, u_de_mean)
        # sigma_omega_omega: [B, C, 2, 2]
        sigma_omega_omega = torch.einsum(
            'b i j, b i j k l -> b i k l', img_preds_1d_spatial_normalized, u_demean_u_demean_T)

        # sigma_omega_omega is symmetric
        # sigma_omega_omega == sigma_omega_omega.permute(0, 1, 3, 2)
        # new torch version has kwarg `left` that we could use if updates
        # A_star: [B, C, 2, 2]
        # Explicitly convert to FP32 since FP16 is not implented for this function (FP16 is possible from `bmm`)
        A_star = torch.linalg.solve(sigma_omega_omega.float(
        ), sigma_F_omega.permute(0, 1, 3, 2).float()).permute(0, 1, 3, 2)

        # F_pred2 = torch.einsum('b i j k, b i l k -> b i l j', A_star, u_de_mean) + mu_F[:, :, None, ...]
        # Do not add mu_F since it's the role for flow_agg
        # F_pred_demean: [B, C, H * W, 2]
        # ([B, C, 2, 2], [B, C, H * W, 2] -> [B, C, H * W, 2])
        F_pred_demean = torch.einsum(
            'b i j k, b i l k -> b i l j', A_star, u_de_mean)

        # F_pred2_2d: [B, C, H, W, 2]
        F_pred2_2d = F_pred_demean.view((B, C, H, W, 2))

        # [B, C, H, W], [B, C, H, W, 2] -> [B, 2, H, W]
        F_pred2_sum_2d = torch.einsum(
            'b i j k, b i j k l -> b l j k', mask, F_pred2_2d)

        return F_pred2_sum_2d

    def aggregate_flow_with_residual(self, mask, flow, all_pred_residual):
        # Two ways:
        # mask1 with fw_flow
        # mask2 with bw_flow
        # mask are expected to sum up to 1 across channel dimension (C=5) before pooling
        B, C, H, W = mask.shape
        # Normalize the mask spatially
        mask_spatial_normalized = mask / \
            mask.view(B, C, H * W, 1).sum(dim=2, keepdim=True)

        # flow before aggregation: [B, C=2, H, W]
        flow_agg = self.flow_feat_before_agg(flow)
        assert flow_agg.shape[2:] == mask_spatial_normalized.shape[2:
                                                                   ], f"{flow_agg.shape[2:]} != {mask_spatial_normalized.shape[2:]} (should match on spatial dimension)"
        # flow_agg[:, :, None, ...]: [B, C=64, 1, H, W]
        # mask_spatial_normalized[:, None, ...]: [B, 1, C=5, H, W]
        flow_agg = flow_agg[:, :, None, ...] * \
            mask_spatial_normalized[:, None, ...]
        # flow_agg after view: [B, C1=64, C2=5, H*W]; after sum: [B, C1=64, C2=5]
        # Equivalent to the form with flatten:
        # flow_agg = flow_agg.view((*flow_agg.shape[:3], -1)).sum(dim=-1)
        flow_agg = flow_agg.flatten(3, 4).sum(dim=-1)
        # After `flow_feat_after_agg`: [B, C1=2, C2=5]
        flow_agg = self.flow_feat_after_agg(flow_agg)
        # After: [B, C1=2, C2=5, 1, 1]
        flow_agg = flow_agg[..., None, None]
        # mask[:, None, ...] : [B, 1, C2=5, H, W]
        # After (flow_agg): [B, C1=2, C2=5, H, W]
        flow_agg = flow_agg * mask[:, None, ...]
        # After (sum up the flows): [B, C1=2, H, W]
        flow_agg = flow_agg.sum(dim=2)

        flow_affine = None
        if self.free_residual:
            # Residual without the constraint of residual map
            # all_pred_residual (before unflatten): [B, 2*5=10, H, W]
            if self.allow_residual_resize and all_pred_residual.shape[-2:] != self.mask_size:
                all_pred_residual = F.interpolate(
                    all_pred_residual, self.mask_size, mode='bilinear')
            all_pred_residual = all_pred_residual.unflatten(
                1, (2, self.mask_layer))
            # mask[:, None, ...]: [B, 1, C2=5, H, W]
            # all_pred_residual:     [B, 2, C2=5, H, W]
            # residual_adjustment:   [B, C1=2, H, W]
            if self.residual_adjustment_scale != -1.:
                residual_adjustment = (torch.tanh(all_pred_residual / self.pred_div_coeff)
                                       * mask[:, None, ...]).sum(dim=2) * self.residual_adjustment_scale
            else:
                # print("Free residual without a limit")
                # Dividing by 10 and multiplying by 10 cancels out
                residual_adjustment = (
                    all_pred_residual * mask[:, None, ...]).sum(dim=2)
                # print(residual_adjustment)
            flow_overall = flow_agg + residual_adjustment
        elif self.free_residual_with_affine:
            flow_affine = self.get_demean_affine_flow(mask, flow)

            # Residual without the constraint of residual map
            # all_pred_residual (before unflatten): [B, 2*5=10, H, W]
            if self.allow_residual_resize and all_pred_residual.shape[-2:] != self.mask_size:
                all_pred_residual = F.interpolate(
                    all_pred_residual, self.mask_size, mode='bilinear')
            all_pred_residual = all_pred_residual.unflatten(
                1, (2, self.mask_layer))
            # mask[:, None, ...]: [B, 1, C2=5, H, W]
            # all_pred_residual:     [B, 2, C2=5, H, W]
            # residual_adjustment:   [B, C1=2, H, W]
            residual_adjustment = (torch.tanh(all_pred_residual / self.pred_div_coeff)
                                   * mask[:, None, ...]).sum(dim=2) * self.residual_adjustment_scale
            flow_overall = flow_agg + flow_affine + residual_adjustment
        else:
            # No residual
            flow_overall = flow_agg

        # Output: [B, C1=2, H, W]
        return flow_overall, flow_agg, residual_adjustment, flow_affine

##### topk + disparity    
    def calculate_angle(self, v1, v2):
        unit_v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
        unit_v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
        dot_product = torch.sum(unit_v1 * unit_v2, dim=-1)
        angle = torch.acos(dot_product)
        return angle

    def detect_flow_changes_batch(self, flow_data, threshold=math.pi / 3, dilation_size=7):
        angle_data = torch.atan2(flow_data[:, 1], flow_data[:, 0])

        B, _, H, W = flow_data.shape
        mask = torch.zeros_like(angle_data)
        padded_angle_data = F.pad(angle_data, (1, 1, 1, 1), mode='replicate')

        # Modify the index here
        diff_left = padded_angle_data[:, 1:-1, 1:-1] - padded_angle_data[:, 1:-1, :-2]
        diff_right = padded_angle_data[:, 1:-1, 1:-1] - padded_angle_data[:, 1:-1, 2:]
        diff_up = padded_angle_data[:, 1:-1, 1:-1] - padded_angle_data[:, :-2, 1:-1]
        diff_down = padded_angle_data[:, 1:-1, 1:-1] - padded_angle_data[:, 2:, 1:-1]

        max_diff = torch.max(torch.max(torch.abs(diff_left), torch.abs(diff_right)),
                            torch.max(torch.abs(diff_up), torch.abs(diff_down)))
        mask = max_diff > threshold

        mask_4d = mask.unsqueeze(1).float()  # shape: [B, 1, H, W]

        dilation_kernel = torch.ones(1, 1, dilation_size, dilation_size, device=flow_data.device)

        dilated_mask = F.conv2d(mask_4d, dilation_kernel, padding=dilation_size//2) > 0

        return dilated_mask

    
    def forward(self, imgs, masks, gt_fw_flows, gt_bw_flows, all_pred_residual_fw, all_pred_residual_bw):
        # masks [B, C=5, H, W]: expected to sum up to 1 across channel dimension (C=5)
        # gt_fw_flows: [B, im_num - 1, C=2, H, W]
        # gt_bw_flows: [B, im_num - 1, C=2, H, W]

        flow_loss = {'seg_fw': 0., 'seg_bw': 0.}

        flows = {'gt_flow': [], 'pred_flow': [], 'agg_flow': [],
                 'residual_adj': [], 'affine_flow': []}

        batch_size, im_num, _, im_h, im_w = imgs.shape
        # im_num: number of images in "pairs" (usually 2)
        assert im_num == 2, "Other im_num not implemented"

        
        individual_losses_fw = []
        individual_losses_bw = []
        
        
        
        for i in range(1, im_num):
            # Now we match the flow for simplicity.

            # masks: [B, 2, C=5, H, W]
            # mask1: [B, C=5, H, W]
            mask1 = masks[:, i-1, :, :, :]
            mask2 = masks[:, i, :, :, :]

            # Same index in fw and bw correspond to same frames (just fw and bw)
            # index i-1 corresponds to the flow between mask1 and mask2
            # gt_fw_flow, gt_bw_flow: [B, C=2, H, W]
            gt_fw_flow = gt_fw_flows[:, i-1, ...]
            gt_bw_flow = gt_bw_flows[:, i-1, ...]

            gt_fw_flow = self.norm_and_clamp_flow(gt_fw_flow)
            gt_bw_flow = self.norm_and_clamp_flow(gt_bw_flow)

            # fw_flow_overall, bw_flow_overall: [B, C=2, H, W]
            fw_flow_overall, fw_flow_agg, fw_residual_adjustment, fw_flow_affine = self.aggregate_flow_with_residual(
                mask1, gt_fw_flow, all_pred_residual_fw)
            bw_flow_overall, bw_flow_agg, bw_residual_adjustment, bw_flow_affine = self.aggregate_flow_with_residual(
                mask2, gt_bw_flow, all_pred_residual_bw)


            mask_fw_flow = self.detect_flow_changes_batch(gt_fw_flow)
            mask_bw_flow = self.detect_flow_changes_batch(gt_bw_flow)


            # loss
            if not self.outlier_robust_loss:
                losses_fw = ((gt_fw_flow - fw_flow_overall).abs())*mask_fw_flow
                losses_fw = losses_fw.sum(dim=(1,2,3))/(mask_fw_flow.sum(dim=(1,2,3))+1e-6)
                losses_bw = ((gt_bw_flow - bw_flow_overall).abs())*mask_bw_flow
                losses_bw = losses_bw.sum(dim=(1,2,3))/(mask_bw_flow.sum(dim=(1,2,3))+1e-6)
            else:
                losses_fw = ((((gt_fw_flow - fw_flow_overall).abs()).view(batch_size, -1) + self.eps) ** self.q).mean(dim=1)
                losses_bw = ((((gt_bw_flow - bw_flow_overall).abs()).view(batch_size, -1) + self.eps) ** self.q).mean(dim=1)


            individual_losses_fw.append(losses_fw)
            individual_losses_bw.append(losses_bw)




            _h, _w, flow, flow2 = get_norm_flow(
                lis1=gt_fw_flow, lis2=gt_bw_flow)
            # After cat: [B, C=4, H, W]
            flows['gt_flow'].append(torch.cat([flow, flow2], dim=1))

            _h, _w, flow, flow2 = get_norm_flow(
                lis1=fw_flow_overall, lis2=bw_flow_overall)
            # After cat: [B, C=4, H, W]
            flows['pred_flow'].append(torch.cat([flow, flow2], dim=1))

            _h, _w, flow, flow2 = get_norm_flow(
                lis1=fw_flow_agg, lis2=bw_flow_agg)
            # After cat: [B, C=4, H, W]
            flows['agg_flow'].append(torch.cat([flow, flow2], dim=1))

            _h, _w, flow, flow2 = get_norm_flow(
                lis1=fw_residual_adjustment, lis2=bw_residual_adjustment)
            # After cat: [B, C=4, H, W]
            flows['residual_adj'].append(torch.cat([flow, flow2], dim=1))

            if fw_flow_affine is not None:
                # `fw_flow_affine` is not None means `bw_flow_affine` is not None
                _h, _w, flow, flow2 = get_norm_flow(
                    lis1=fw_flow_affine, lis2=bw_flow_affine)
                # After cat: [B, C=4, H, W]
                flows['affine_flow'].append(torch.cat([flow, flow2], dim=1))




        # Combined and sorted losses topk
        total_losses_fw = torch.cat(individual_losses_fw)
        total_losses_bw = torch.cat(individual_losses_bw)
        total_losses = total_losses_fw + total_losses_bw

        sorted_losses, sorted_indices = torch.sort(total_losses)

        # Select the 2 images with the least loss
        selected_indices = sorted_indices[:2]

        # Calculate the average loss of the selected images
        selected_flow_loss = {
            'seg_fw': total_losses_fw[selected_indices].mean(),
            'seg_bw': total_losses_bw[selected_indices].mean()
        }
        selected_flow_loss['seg'] = selected_flow_loss['seg_fw'] + selected_flow_loss['seg_bw']

        # Update flows and flow_loss to include data for selected images
        # Make sure the length of the list is at least the same as the length of selected_indices
        selected_flows = {}
        for key, value in flows.items():
                # Check the length of each list to make sure it is not out of range
                if len(value) >= len(selected_indices):
                    selected_flows[key] = [value[i] for i in selected_indices]
                else:
                    # If the list is not long enough, copy the entire list
                    selected_flows[key] = value.copy()
                    
        return selected_flows, selected_flow_loss