import torch
import torch.nn as nn
from .clip import clip
import torchvision
from PIL import Image

model_name = "ViT-B/16"

import torchvision.utils as vutils


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


class CLIPEncoder(nn.Module):
    def __init__(self, need_ref=False, ref_path=None):
        super().__init__()
        self.clip_model = load_clip_to_cpu()
        self.clip_model.requires_grad = False
        self.preprocess = torchvision.transforms.Normalize(
            (0.48145466 * 2 - 1, 0.4578275 * 2 - 1, 0.40821073 * 2 - 1),
            (0.26862954 * 2, 0.26130258 * 2, 0.27577711 * 2),
        )
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

            img = Image.open(ref_path).convert("RGB")
            image = img.resize((224, 224), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            self.ref = img

        self.debug = True

    def get_gram_matrix_residual(self, im1):
        im1 = torch.nn.functional.interpolate(im1, size=(224, 224), mode="bicubic")
        im1 = self.preprocess(im1)

        if self.debug:
            x_norm = self.ref * 0.26 + 0.45
            vutils.save_image(x_norm, "ref_cropped_img.png")
            x_norm = (im1 + 1.0) / 2.0
            vutils.save_image(x_norm, "content_cropped_img.png")

        f1, feats1 = self.clip_model.encode_image_with_features(im1)
        f2, feats2 = self.clip_model.encode_image_with_features(self.ref)

        feat1 = feats1[2][1:, 0, :]
        feat2 = feats2[2][1:, 0, :]
        gram1 = torch.mm(feat1.t(), feat1)
        gram2 = torch.mm(feat2.t(), feat2)
        return gram1 - gram2

    def forward(self, image, text):
        im1 = torch.nn.functional.interpolate(image, size=(224, 224), mode="bicubic")
        im1 = self.preprocess(im1)
        text = clip.tokenize([text]).cuda()
        return self.clip_model.forward(im1, text)


if __name__ == "__main__":
    m = CLIPEncoder().cuda()
    im1 = torch.randn((1, 3, 224, 224)).cuda()
    im2 = torch.randn((1, 3, 224, 224)).cuda()
    m.get_gram_matrix_residual(im1, im2)
