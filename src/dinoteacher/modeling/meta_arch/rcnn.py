from typing import Tuple, Optional
import torch.nn as nn
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from adapteacher.modeling.meta_arch.rcnn import DAobjTwoStagePseudoLabGeneralizedRCNN
from typing import Dict, Tuple, List, Optional
import torch
from torchvision.transforms import v2
from detectron2.structures import ImageList

@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN_shortcut(DAobjTwoStagePseudoLabGeneralizedRCNN):
    def __init__(
        self,
        cfg
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(DAobjTwoStagePseudoLabGeneralizedRCNN_shortcut, self).__init__(cfg)

    def preprocess_depth(self, batched_inputs: List[Dict[str, torch.Tensor]], height: int, width: int):
        """
        Normalize, pad and batch the input images.
        """
        resize = v2.Resize((height, width), antialias=True)
        depths = [resize(x["depth"]).to(self.device) for x in batched_inputs]
        # depths = [(x - self.pixel_mean) / self.pixel_std for x in depths]
        depths = ImageList.from_tensors(depths, self.backbone.size_divisibility)

        return depths

    def forward_backbone(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        try:
            depths = self.preprocess_depth(batched_inputs, images.tensor.shape[-2], images.tensor.shape[-1])
            depths = depths.tensor[:, 0:1, :, :].float()
            depths = (depths - depths.amin((2,3), True)) / (depths.amax((2,3), True) - depths.amin((2,3), True) + 1e-8)
            features = self.backbone(images.tensor * (1 + depths))
        except:
            features = self.backbone(images.tensor)
        return features