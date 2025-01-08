import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from .init_weight import BaseModel
from .head import SegFormerHead
from src.registry import MODEL


@MODEL.register("segformer")
class SegFormer(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__(backbone, num_classes)
        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, num_classes=150)
        # self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, num_classes) 
        self.apply(self._init_weights)

        pretrained_path_dict = {
            'MiT-B1': 'pretrained/segformer.b1.ade.pth',
            'MiT-B2': 'pretrained/segformer.b2.ade.pth',
            'MiT-B3': 'pretrained/s.egformer.b3.ade.pth'
        }

        msg = self.load_state_dict(torch.load(pretrained_path_dict[backbone], map_location='cpu'))
        print(msg)

        self.decode_head.linear_pred = nn.Conv2d(
            self.decode_head.linear_pred.in_channels,
            num_classes, 
            kernel_size=(1, 1)
        )


    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y


if __name__ == '__main__':
    model = SegFormer('MiT-B0')
    x = torch.zeros(1, 3, 512, 512)
    y = model(x)
    print(y.shape)