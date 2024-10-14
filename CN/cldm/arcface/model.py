import torch
from torch import nn
from .facial_recognition.model_irse import Backbone
import torchvision
import torchvision.utils as vutils
from PIL import Image


class IDLoss(nn.Module):
    def __init__(self, ref_path=None, ref_2nd_path=None):
        super(IDLoss, self).__init__()
#         print('Loading ResNet ArcFace for ID Loss')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load("/ssd/model/arcface/model_ir_se50.pth"))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.to_tensor = torchvision.transforms.ToTensor()

        self.ref_path = "/workspace/ddgm/functions/arcface/land.png" if not ref_path else ref_path
        self.ref_2nd_path = "/home/hexiangyu/FreeDoM/CN/test_imgs/id7.jpg" if not ref_2nd_path else ref_2nd_path
        img = Image.open(self.ref_path)
        img = img.resize((256, 256), Image.BILINEAR)
        img = self.to_tensor(img)
        self.ref = torch.unsqueeze(img * 2.0 - 1, 0).cuda()
        img_ref_2nd = Image.open(self.ref_2nd_path)
        img_ref_2nd = img_ref_2nd.resize((256, 256), Image.BILINEAR)
        img_ref_2nd = self.to_tensor(img_ref_2nd)
        self.ref_2nd = torch.unsqueeze(img_ref_2nd * 2.0 - 1, 0).cuda()
        self.debug = True

    def extract_feats(self, x, prefix):
        if x.shape[2] != 256:
            x = self.pool(x)
        # x = x[:, :, 35:223, 32:220]  # Crop interesting region
        # if 'input' in prefix:
        #     x = x[:, :, 35:223, 32:220]  # Crop interesting region
        if self.debug:
            x_norm = (x + 1.0) / 2.0
            vutils.save_image(x_norm, prefix + '_cropped_face_img.png')
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats
    
    def extract_feats_not_align(self, x):
        # if x.shape[2] != 256:
        #     x = self.pool(x)
        # x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def get_residual(self, image, dual_sup=False):
        if dual_sup:
            img_feat = self.extract_feats(image, 'second_input')
            ref_feat = self.extract_feats(self.ref_2nd, 'second_ref')
        else:
            img_feat = self.extract_feats(image, 'first_input')
            ref_feat = self.extract_feats(self.ref, 'first_ref')
        return ref_feat - img_feat







