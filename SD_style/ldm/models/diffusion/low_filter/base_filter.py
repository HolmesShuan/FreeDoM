import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torchvision
from PIL import Image
from .clip import clip
from .utils import load_image, save_image, gram_matrix, normalize_batch
from .vgg import Vgg16

from torchvision import transforms

import torchvision.utils as vutils


model_name = "ViT-B/16"


def load_clip_to_cpu():
    url = clip._MODELS[model_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class VGG16Encoder(nn.Module):
    def __init__(
        self, need_ref=False, ref_path=None, ref_mask_path=None, image_size=224
    ):
        super().__init__()
        self.vgg_model = Vgg16(requires_grad=False).cuda()
        self.vgg_model.requires_grad = False
        self.image_size = image_size
        self.debug = True

        if need_ref:
            style_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))]
            )
            self.ref_path = ref_path
            self.ref_mask_path = ref_mask_path

            style = load_image(self.ref_path, size=self.image_size)
            self.ref = style_transform(style).repeat(1, 1, 1, 1).cuda()

            if self.ref_mask_path is not None:
                np_array = np.load(self.ref_mask_path)
                torch_array = (
                    torch.from_numpy(np_array).cuda().unsqueeze(0).unsqueeze(0)
                )
                self.mask_tensor = F.interpolate(
                    torch_array, size=style.size, mode="nearest"
                )
                features_ref = self.vgg_model(
                    normalize_batch(self.ref * self.mask_tensor)
                )
            else:
                features_ref = self.vgg_model(normalize_batch(self.ref))
            self.content_features_ref = features_ref.relu2_2
            self.gram_style_ref = [gram_matrix(y) for y in features_ref]

    def get_content_residual(self, im1):
        im1 = F.interpolate(
            im1, size=(self.image_size, self.image_size), mode="bicubic"
        )
        # im1 = torch.nn.functional.interpolate(im1, size=(512, 512), mode='bicubic')
        im1 = (im1 + 1.0) / 2.0 * 255
        if self.ref_mask_path is not None:
            x = normalize_batch(im1 * self.mask_tensor)
        else:
            x = normalize_batch(im1)
        features_x = self.vgg_model(x)

        if self.debug:
            x_norm = self.ref / 255.0
            vutils.save_image(x_norm, "ref_cropped_img.png")
            x_norm = im1 / 255.0
            vutils.save_image(x_norm, "content_cropped_img.png")

        return features_x.relu2_2 - self.content_features_ref

    def get_gram_matrix_style_loss(self, im1):
        im1 = torch.nn.functional.interpolate(
            im1, size=(self.image_size, self.image_size), mode="bicubic"
        )
        # im1 = torch.nn.functional.interpolate(im1, size=(512, 512), mode='bicubic')
        im1 = (im1 + 1.0) / 2.0 * 255
        if self.ref_mask_path is not None:
            x = normalize_batch(im1 * self.mask_tensor)
        else:
            x = normalize_batch(im1)
        features_x = self.vgg_model(x)

        if self.debug:
            x_norm = self.ref / 255.0
            vutils.save_image(x_norm, "ref_cropped_img.png")
            x_norm = im1 / 255.0
            vutils.save_image(x_norm, "content_cropped_img.png")

        style_loss = 0.0
        for ft_y, gm_s in zip(features_x, self.gram_style_ref):
            gm_y = gram_matrix(ft_y)
            style_loss += torch.linalg.norm(gm_y - gm_s[:1, :, :])

        return style_loss


class LowFilter(nn.Module):
    def __init__(self, need_ref=True, ref_path=None, ref_mask_path=None):
        super().__init__()
        self.preprocess = torchvision.transforms.Normalize(
            (0.48145466 * 2 - 1, 0.4578275 * 2 - 1, 0.40821073 * 2 - 1),
            (0.26862954 * 2, 0.26130258 * 2, 0.27577711 * 2),
        )
        self.use_nn = False
        self.debug = True
        if need_ref:
            self.to_tensor = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
            self.ref_path = (
                ref_path
                if ref_path is not None
                else "/home/hexiangyu/FreeDoM/SD_style/content_images/zuanshijian.png"
            )
            self.ref_mask_path = (
                ref_mask_path
                if ref_mask_path is not None
                else "/home/hexiangyu/FreeDoM/SD_style/content_images/zuanshijian.npy"
            )
            img = Image.open(self.ref_path).convert("RGB")
            width, height = img.size
            if width > height:
                self.new_width = 512
                self.new_height = int(height * (self.new_width / width))
            else:
                self.new_height = 512
                self.new_width = int(width * (self.new_height / height))
            if self.use_nn:
                self.new_height = 512
                self.new_width = 512
            image = img.resize((self.new_width, self.new_height), Image.BILINEAR)
            self.ref = self.to_tensor(image).unsqueeze(0).cuda()
            np_array = np.load(self.ref_mask_path)
            torch_array = torch.from_numpy(np_array).cuda().unsqueeze(0).unsqueeze(0)
            self.mask_tensor = F.interpolate(
                torch_array, size=(self.new_width, self.new_height), mode="nearest"
            )
        if self.use_nn:
            self.clip_model = load_clip_to_cpu()
            self.clip_model.requires_grad = True

    def center_crop(self, tensor, target_size):
        _, _, height, width = tensor.shape
        crop_height, crop_width = target_size

        start_y = (height - crop_height) // 2
        start_x = (width - crop_width) // 2

        return tensor[
            :, :, start_y : start_y + crop_height, start_x : start_x + crop_width
        ]

    def get_low_fass_filter_residual(self, im1):
        # im1 = torch.nn.functional.interpolate(im1, size=(224, 224), mode='bicubic')
        im1 = self.center_crop(im1, (self.new_width, self.new_height))
        im1 = self.preprocess(im1)
        if self.use_nn:
            ref_image_features = self.clip_model.encode_image(
                self.ref * self.mask_tensor
            )
            content_image_features = self.clip_model.encode_image(
                im1 * self.mask_tensor
            )
            # normalized features
            ref_image_features = ref_image_features / ref_image_features.norm(
                dim=-1, keepdim=True
            )
            content_image_features = (
                content_image_features
                / content_image_features.norm(dim=-1, keepdim=True)
            )
            if self.debug:
                x_norm = (self.ref + 1.0) / 2.0
                vutils.save_image(x_norm, "ref_cropped_face_img.png")
                x_norm = (im1 + 1.0) / 2.0
                vutils.save_image(x_norm, "content_cropped_face_img.png")
            return ref_image_features - content_image_features
        else:
            ref_image_features = F.interpolate(
                F.interpolate(
                    im1,
                    size=(self.new_width // 4, self.new_height // 4),
                    mode="bicubic",
                ),
                size=(self.new_width, self.new_height),
                mode="bicubic",
            )
            content_image_features = F.interpolate(
                F.interpolate(
                    self.ref,
                    size=(self.new_width // 4, self.new_height // 4),
                    mode="bicubic",
                ),
                size=(self.new_width, self.new_height),
                mode="bicubic",
            )
            if self.debug:
                x_norm = (ref_image_features + 1.0) / 2.0
                vutils.save_image(x_norm, "ref_cropped_face_img.png")
                x_norm = (content_image_features + 1.0) / 2.0
                vutils.save_image(x_norm, "content_cropped_face_img.png")
            return (ref_image_features - content_image_features) * self.mask_tensor
