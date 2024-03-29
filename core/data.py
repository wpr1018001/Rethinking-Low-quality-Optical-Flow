import torch
import torch.utils.data
import numpy as np
import os
from PIL import Image
import random

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, training, frame_num=2, load_flow=False, load_pl=False, transform=None, subsample_frame_interval=None, flow_suffix="",flow_suffix2="",flow_suffix3="", zero_ann=False, pl_root=None, pl_root2=None):
        super().__init__()

        file_path = os.path.join(root, split)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        lines.sort()
        seq_lens = []
        seq_names = []
        seq_frames_path_all = []
        if subsample_frame_interval is not None:
            assert not training, "subsample_frame_interval is only for evaluation"
        for line in lines:
            line = line.split()
            seq_name = line[0]
            seq_frames = line[1:]
            if subsample_frame_interval == -1:
                # First frame for every sequence
                seq_frames = seq_frames[:1]
            elif subsample_frame_interval is not None:
                seq_frames = seq_frames[::subsample_frame_interval]
            seq_lens.append(len(seq_frames))
            seq_names.append(seq_name.rstrip("/").split("/")[-1])
            seq_frames_path_all.append(
                [os.path.join(root, seq_name, frame_filename) for frame_filename in seq_frames])

        self.seq_names = seq_names
        self.seq_frames_path_all = seq_frames_path_all

        self.seq_lens = seq_lens
        self.seq_freq = seq_lens / np.sum(seq_lens)
        self.seq_len_cumsum = np.insert(np.cumsum(seq_lens), 0, 0)
        # print(self.seq_len_cumsum)
        self.num_seq = len(seq_lens)

        self.transform = transform

        self.frame_num = frame_num
        self.training = training
        self.load_flow = load_flow
        self.load_pl = load_pl
        self.flow_suffix = flow_suffix
        self.flow_suffix2 = flow_suffix2
        self.flow_suffix3 = flow_suffix3
        self.pl_root = pl_root
        self.pl_root2 = pl_root2

        self.zero_ann = zero_ann

        if self.load_pl:
            assert self.transform.has_pl, "load_pl needs to match with has_pl in transform"

        if not self.training:
            assert self.frame_num == 1, f"You need single frames for evaluaion but have {self.frame_num} frames"

    def load_image(self, path, convert_format="RGB"):
        with open(path, "rb") as f:
            try:
                img = Image.open(f)
            except Exception as e:
                print("Error in loading image: ", e)
                img = Image.open(f)
            return img.convert(convert_format)

    def __getitem__(self, index):
        # subset is taken in the split txt file
        seq_ind_within_subset = np.digitize(index, self.seq_len_cumsum) - 1

        frame_ind = index - self.seq_len_cumsum[seq_ind_within_subset]

        # We don't get the last `self.frame_num - 1` frame(s) since we need current and next frame
        if frame_ind >= self.seq_lens[seq_ind_within_subset] - (self.frame_num - 1):
            frame_ind -= self.frame_num - 1
            assert self.training, f"In evaluation, we should use single frame to evaluate. Index: {index}, frame index: {frame_ind}."

        current_seq = self.seq_frames_path_all[seq_ind_within_subset]

        # images = []
        # for i in range(self.frame_num):
        #     path = current_seq[frame_ind + i]
        #     image = self.load_image(path)
        #     images.append(image)
        
        # Randomly select whether to acquire the next frame or two frames apart
        options = [1, 2, 3]
        probabilities = [0.7, 0.2, 0.1]
        frame_gap = np.random.choice(options, p=probabilities)  # 1 means next frame, 2 means two frames apart.

        images = []
        flag_gap = 0
        for i in range(self.frame_num):
            # Calculate the index of the frame to be acquired
            frame_to_get = frame_ind + i * frame_gap

            # Make sure the index is within legal limits
            # If the requirement of two frames apart cannot be met, adjacent frames are used
            if frame_to_get >= len(current_seq):
                frame_to_get = frame_ind + i  # Using adjacent frames
                flag_gap = 1
            else:
                frame_to_get = min(frame_to_get, len(current_seq) - 1)  # Using Interval Frames
                flag_gap = frame_gap

            path = current_seq[frame_to_get]
            image = self.load_image(path)
            images.append(image)

        seq_name = self.seq_names[seq_ind_within_subset]

        ret = {
            'imgs': images, 
            'seq_ids': seq_ind_within_subset,
            'seq_names': seq_name, 
            'paths': current_seq[frame_ind:frame_ind+self.frame_num], 
            'frame_ind_start': frame_ind, 
            'seg_fields': []
        }

        if not self.training:
            # Assume we have only one frame
            assert i == 0, "In eval, we should have one frame only."
            if not self.zero_ann:
                path = current_seq[frame_ind].replace(
                    "JPEGImages", "Annotations").replace(".png", ".jpg")    # rewrite by wpr
                ann = self.load_image(path)
            else:
                # Set ann to 1x1 zeros
                ann = Image.fromarray(np.array([[[0, 0, 0]]], dtype=np.uint8))
            # Do not resize annotations (not adding into seg_fields): mask will be resized to annotations
            # Support one annotation for now
            ret['ann'] = ann

        if self.load_flow:
            if flag_gap == 1:
                gt_fw_flows = []
                gt_bw_flows = []
                for i in range(1, self.frame_num): # 00001.jpg in Flow is the flow from 0 to 1
                    fw_flow_path = current_seq[frame_ind + i].replace(
                        "JPEGImages", "Flows" + self.flow_suffix)[:-4] + ".npy"
                    bw_flow_path = current_seq[frame_ind + i].replace(
                        "JPEGImages", "BackwardFlows" + self.flow_suffix)[:-4] + ".npy"
                    if False: # debug
                        fw_flow_path = "/home/l/lo/longlian/00001.npy"
                        bw_flow_path = "/home/l/lo/longlian/00001.npy"
                    gt_fw_flow = np.load(fw_flow_path)
                    gt_bw_flow = np.load(bw_flow_path)

                    ### Data format modification
                    gt_fw_flow = gt_fw_flow.astype(np.float32)
                    gt_bw_flow = gt_bw_flow.astype(np.float32)
                    
                    gt_fw_flows.append(gt_fw_flow)
                    gt_bw_flows.append(gt_bw_flow)
            elif flag_gap == 2:
                gt_fw_flows = []
                gt_bw_flows = []
                for i in range(1, self.frame_num): # 00001.jpg in Flow is the flow from 0 to 1
                    fw_flow_path = current_seq[frame_ind + i].replace(
                        "JPEGImages", "Flows" + self.flow_suffix2)[:-4] + ".npy"
                    bw_flow_path = current_seq[frame_ind + i].replace(
                        "JPEGImages", "BackwardFlows" + self.flow_suffix2)[:-4] + ".npy"
                    if False: # debug
                        fw_flow_path = "/home/l/lo/longlian/00001.npy"
                        bw_flow_path = "/home/l/lo/longlian/00001.npy"
                    gt_fw_flow = np.load(fw_flow_path)
                    gt_bw_flow = np.load(bw_flow_path)

                    ### Data format modification
                    gt_fw_flow = gt_fw_flow.astype(np.float32)
                    gt_bw_flow = gt_bw_flow.astype(np.float32)
                    
                    gt_fw_flows.append(gt_fw_flow)
                    gt_bw_flows.append(gt_bw_flow)
            elif flag_gap == 3:
                gt_fw_flows = []
                gt_bw_flows = []
                for i in range(1, self.frame_num): # 00001.jpg in Flow is the flow from 0 to 1
                    fw_flow_path = current_seq[frame_ind + i].replace(
                        "JPEGImages", "Flows" + self.flow_suffix3)[:-4] + ".npy"
                    bw_flow_path = current_seq[frame_ind + i].replace(
                        "JPEGImages", "BackwardFlows" + self.flow_suffix3)[:-4] + ".npy"
                    if False: # debug
                        fw_flow_path = "/home/l/lo/longlian/00001.npy"
                        bw_flow_path = "/home/l/lo/longlian/00001.npy"
                    gt_fw_flow = np.load(fw_flow_path)
                    gt_bw_flow = np.load(bw_flow_path)

                    ### Data format modification
                    gt_fw_flow = gt_fw_flow.astype(np.float32)
                    gt_bw_flow = gt_bw_flow.astype(np.float32)
                    
                    gt_fw_flows.append(gt_fw_flow)
                    gt_bw_flows.append(gt_bw_flow)

            ret['gt_fw_flows'] = gt_fw_flows
            ret['gt_bw_flows'] = gt_bw_flows
            ret['seg_fields'].extend(['gt_fw_flows', 'gt_bw_flows'])
            

        if self.load_pl:
            # PL is different from annotaiton as it requires augmentation
            
            
            # pl_masks_1
            pl_masks_1 = []
            for i in range(self.frame_num):
                # frame_ind + i
                img_filename = current_seq[frame_ind + i].split('/')[-1][:-4]
                path = os.path.join(self.pl_root, f'pred_seg_{seq_name}_{img_filename}_0000000.png')
                pl_mask_1 = np.asarray(self.load_image(path, convert_format="L"))
                pl_masks_1.append(pl_mask_1)
            ret['pl_masks_1'] = pl_masks_1
            ret['seg_fields'].append('pl_masks_1')
            
            # pl_masks_2
            pl_masks_2 = []
            for j in range(self.frame_num):
                # frame_ind + j
                img_filename2 = current_seq[frame_ind + j].split('/')[-1][:-4]
                path2 = os.path.join(self.pl_root2, f'pred_seg_{seq_name}_{img_filename2}_0000000.png')
                pl_mask_2 = np.asarray(self.load_image(path2, convert_format="L"))
                pl_masks_2.append(pl_mask_2)
            ret['pl_masks_2'] = pl_masks_2
            ret['seg_fields'].append('pl_masks_2')
                                   

        if self.transform is not None:
            ret = self.transform(ret)

        # When collated, imgs will become `self.frame_num` arrays. Same for paths.
        return ret

    def __len__(self):
        # Debugging:
        # return 8
        return np.sum(self.seq_lens)


if __name__ == "__main__":
    np.random.seed(1)
    
    dataset = VideoDataset('../data/data_SegTrackv2_resized', training=True, load_flow=True, split='trainval.txt', flow_suffix="_NewCT")

    for item in dataset:
        continue

