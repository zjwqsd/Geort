# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os 
from pathlib import Path
from geort.formatter import HandFormatter
from geort.model import IKModel
from geort.utils.path import to_package_root, get_checkpoint_root
from geort.utils.config_utils import load_json, parse_config_keypoint_info, parse_config_joint_limit


class GeoRTRetargetingModel:
    '''
        Used by external programs.
    '''
    def __init__(self, model_path, config_path,device=None):
        config = load_json(config_path)
        keypoint_info = parse_config_keypoint_info(config)
        joint_lower_limit, joint_upper_limit = parse_config_joint_limit(config)
        print(keypoint_info["joint"])
        self.human_ids = keypoint_info["human_id"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = IKModel(keypoint_joints=keypoint_info["joint"]).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.qpos_normalizer = HandFormatter(joint_lower_limit, joint_upper_limit) # GeoRT will do normalization.

    def forward(self, keypoints):
        # keypoints: [N, 3]
        keypoints = keypoints[self.human_ids] # extract.
        # joint_normalized = self.model.forward(torch.from_numpy(keypoints).unsqueeze(0).reshape(1, -1, 3).float().cuda())
        input_tensor = torch.from_numpy(keypoints).unsqueeze(0).reshape(1, -1, 3).float().to(self.device)
        joint_normalized = self.model.forward(input_tensor)
        joint_raw = self.qpos_normalizer.unnormalize(joint_normalized.detach().cpu().numpy())
        return joint_raw[0]


def load_model(tag='', epoch=0,device=None):
    '''
        Loading API.
    '''
    checkpoint_root = get_checkpoint_root()
    all_checkpoints = os.listdir(checkpoint_root)
    
    checkpoint_name = ''
    for checkpoint in all_checkpoints:
        if tag in checkpoint:
            checkpoint_name = checkpoint
            break 

    checkpoint_root = Path(checkpoint_root) / checkpoint_name
    if epoch > 0:
        model_path = checkpoint_root / f"epoch_{epoch}.pth"
    else:
        model_path = checkpoint_root / f"last.pth"
    
    config_path = checkpoint_root / "config.json"
    return GeoRTRetargetingModel(model_path=model_path, config_path=config_path, device=device)

if __name__ == '__main__':
    # load the model in one line.
    model = load_model(tag="allegro_last", device=torch.device("cpu"))
