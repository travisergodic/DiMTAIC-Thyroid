import torch
import torch.nn.functional as F

from src.registry import TTA


@TTA.register("identity")
class IdentityTTAWrapper:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    @torch.no_grad()
    def __call__(self, data, mask=None):
        if mask is None:
            label = self.model(data)
        else:
            label = self.model(data, mask=mask)
        return label
        # mask = torch.sigmoid(mask)
        # return (mask > self.threshold).long()


@TTA.register("multiscale")
class MultiScaleTTAWrapper:
    def __init__(self, model, scales, task="seg", agg="mean"):
        self.model = model
        self.scales = scales
        self.task = task.lower()
        self.agg = agg.lower()
    
    @torch.no_grad()
    def __call__(self, data, mask=None):
        H, W = data.size()[-2:]
        label_list = []

        for scale in self.scales:
            # resize data
            data = F.interpolate(
                data, 
                size=scale, 
                mode='bilinear', 
                align_corners=False
            )
            # model forward
            if mask is None:
                label = self.model(data)
            else:
                label = self.model(data, mask)

            label = torch.sigmoid(label)

            if self.task == "seg":
                label = F.interpolate(
                    label, 
                    size=(H, W), 
                    mode='nearest'
                )
            label_list.append(label)

        if self.agg == "mean":
            return sum(label_list).div(len(self.scales))
        
        elif self.agg == "max":
            stacked_labels = torch.stack(label_list, dim=1)  # Shape: (B, N, 1)
            max_labels, _ = torch.max(stacked_labels, dim=1)  # Shape: (B, 1)
            return max_labels
        

@TTA.register("cls_multipatch")
class MultiPatchClassificationTTAWrapper:
    def __init__(self, model, patch_size, stride, img_size, agg="max"):
        self.model = model
        self.agg = agg.lower()
        self.patch_size = patch_size
        self.stride = stride
        self.img_size = img_size
    
    @torch.no_grad()
    def __call__(self, data):
        assert data.ndim == 4
        assert data.size(0) == 1

        data = data[0]
        patches = data.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
        patches = patches.contiguous().view(-1, 3, self.patch_size, self.patch_size)

        patches = F.interpolate(
            patches, 
            size=self.img_size, 
            mode='bilinear', 
            align_corners=False
        )
        ori_img = F.interpolate(
            data.unsqueeze(0), 
            size=self.img_size, 
            mode='bilinear', 
            align_corners=False
        )
        input_batch = torch.concat([ori_img, patches], dim=0)
        label = self.model(input_batch)
        if self.agg == "mean":
            res = torch.mean(label)

        elif self.agg == "max":
            res = torch.max(label)
        return res.view(1, 1)
    

@TTA.register("seg_multipatch")
class MultiPatchSegmentationTTAWrapper:
    def __init__(self, model, patch_size, stride, img_size):
        self.model = model
        # self.agg = agg.lower()
        self.patch_size = patch_size
        self.stride = stride
        self.img_size = img_size
    
    @torch.no_grad()
    def __call__(self, data):
        assert data.ndim == 4
        assert data.size(0) == 1  # batch size of 1
        
        data = data[0]  # Remove batch dimension
        c, h, w = data.shape
        
        # Unfold the input image into overlapping patches
        patches = data.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
        num_patches_h = patches.size(1)
        num_patches_w = patches.size(2)
        patches = patches.contiguous().view(-1, c, self.patch_size, self.patch_size)  # Shape: (N_patches, C, patch_size, patch_size)
        
        # Rescale all patches to model input size
        patches = F.interpolate(patches, size=self.img_size, mode='bilinear', align_corners=False)
        
        # Apply the segmentation model to all patches in one batch
        patch_predictions = self.model(patches)  # Shape: (N_patches, C_out, img_size, img_size)
        
        # Rescale each patch prediction back to the patch size
        patch_predictions = F.interpolate(patch_predictions, size=(self.patch_size, self.patch_size), mode='bilinear', align_corners=False)
        
        # Initialize output and count tensors
        output = torch.zeros((1, self.img_size, self.img_size)).to(data.device)
        count = torch.zeros((1, self.img_size, self.img_size)).to(data.device)

        # Reshape predictions back to the spatial layout of patches
        patch_predictions = patch_predictions.view(num_patches_h, num_patches_w, -1, self.patch_size, self.patch_size)
        
        # Add each patch's prediction to the correct location in the output tensor
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                y = i * self.stride
                x = j * self.stride
                output[:, y:y+self.patch_size, x:x+self.patch_size] += patch_predictions[i, j]
                count[:, y:y+self.patch_size, x:x+self.patch_size] += 1

        # Normalize by count to handle overlapping areas
        output /= torch.clamp(count, min=1)
        return output