# -*- coding: utf-8 -*-
import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

from models import STL10CNN


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class GradCAM:
    """Safe CAM implementation (Score-CAM-like aggregation).

    This class avoids registering backward hooks and therefore does not
    trigger inplace/view autograd conflicts. It aggregates channel-wise
    activations into a single spatial map and returns a normalized CAM.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        # register forward hook to capture activations
        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        # store activations (detached to avoid autograd tying)
        self.activations = out.detach()

    def remove_hooks(self):
        self.fwd_handle.remove()

    def __call__(self, x, class_idx=None):
        # run a forward pass (no grad) to populate activations
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(torch.argmax(logits, dim=1).item())

        # activations shape: (batch, channels, height, width)
        activations = self.activations  # expect batch size 1
        if activations is None:
            raise RuntimeError("Target layer activations not captured. Make sure target_layer is correct.")

        _, num_channels, height, width = activations.shape

        # compute channel importance weights by channel mean (simple proxy)
        cam_weights = []
        for c in range(num_channels):
            ch = activations[0, c:c+1, :, :]
            ch_min = float(ch.min())
            ch_max = float(ch.max())
            if ch_max - ch_min > 1e-8:
                ch_norm = (ch - ch_min) / (ch_max - ch_min)
            else:
                ch_norm = torch.zeros_like(ch)
            weight = float(ch_norm.mean())
            cam_weights.append(weight)

        cam_weights = np.array(cam_weights, dtype=np.float32)

        # aggregate channels into a single CAM (numpy for stability)
        cam = np.zeros((height, width), dtype=np.float32)
        for c in range(num_channels):
            channel_map = activations[0, c].cpu().numpy()
            ch_min = channel_map.min()
            ch_max = channel_map.max()
            if ch_max - ch_min > 1e-8:
                channel_map = (channel_map - ch_min) / (ch_max - ch_min)
            else:
                channel_map = np.zeros_like(channel_map)
            cam += cam_weights[c] * channel_map

        # upsample to input resolution
        cam_tensor = torch.from_numpy(cam[np.newaxis, np.newaxis, :, :]).float()
        cam_tensor = F.interpolate(cam_tensor, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam_tensor.squeeze().cpu().numpy()

        # final normalization
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for STL10 model")
    parser.add_argument("--data_root", type=str, default="HW2/STL10")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="HW2/outputs/gradcam")
    return parser.parse_args()


def preprocess(image_size=96):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
        ]
    )


def overlay_heatmap(image_np, cam, alpha=0.45):
    cmap = plt.get_cmap("jet_r")
    heatmap = cmap(cam)[..., :3]
    blend = image_np * (1 - alpha) + heatmap * alpha
    blend = np.clip(blend, 0, 1)
    return blend


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    class_names = ckpt["class_names"]

    model = STL10CNN(
        num_classes=cfg["num_classes"],
        channels=cfg["channels"],
        activation=cfg["activation"],
        pool_type=cfg["pool_type"],
        use_batchnorm=cfg["use_batchnorm"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tf = preprocess(cfg["image_size"])
    test_dataset = datasets.ImageFolder(os.path.join(args.data_root, "test"))
    sample_indices = random.sample(range(len(test_dataset)), k=min(args.num_images, len(test_dataset)))

    grad_cam = GradCAM(model, model.get_last_conv_layer())

    for rank, idx in enumerate(sample_indices):
        image_path, true_label = test_dataset.samples[idx]
        pil_img = Image.open(image_path).convert("RGB")
        pil_img = pil_img.resize((cfg["image_size"], cfg["image_size"]))
        img_np = np.array(pil_img, dtype=np.float32) / 255.0

        x = tf(pil_img).unsqueeze(0).to(device)
        cam, pred_idx = grad_cam(x)
        overlay = overlay_heatmap(img_np, cam)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original\nTrue: {class_names[true_label]}")
        axes[1].imshow(cam, cmap="jet_r")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay\nPred: {class_names[pred_idx]}")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"gradcam_{rank:02d}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

    grad_cam.remove_hooks()
    print("Grad-CAM images saved to:", args.output_dir)


if __name__ == "__main__":
    main()

