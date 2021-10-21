
import cv2
import numpy as np
import torch
from kp2d.networks.keypoint_net import KeypointNet
from kp2d.utils.image import to_color_normalized, to_gray_normalized
from kp2d.evaluation.descriptor_evaluation import select_k_best
import torchvision.transforms as transforms
import argparse


class Kp2dWrapper:

    def __init__(self, model_path, conf_threshold=0.0, k_best=1000):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.k_best = k_best
        self.__load_model()

    def __load_model(self):
        checkpoint = torch.load(self.model_path)
        model_args = checkpoint['config']['model']['params']
        # Create and load disp net
        self.use_color = model_args['use_color']
        keypoint_net = KeypointNet(use_color=model_args['use_color'],
                                   do_upsample=model_args['do_upsample'],
                                   do_cross=model_args['do_cross'])
        keypoint_net.load_state_dict(checkpoint['state_dict'])
        keypoint_net = keypoint_net.cuda()
        keypoint_net.eval()
        keypoint_net.training = False
        self.keypoint_net = keypoint_net

    def __read_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_size = (img.shape[1], img.shape[0])
        if not self.use_color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        with torch.no_grad():
            transform = transforms.ToTensor()
            img = transform(img).type('torch.FloatTensor')
            img = torch.reshape(img, torch.Size([1]) + img.shape)
            if self.use_color:
                img = to_color_normalized(img.cuda())
            else:
                img = to_gray_normalized(img.cuda())
        return img, img_size

    def process_image(self, img_path):
        img, img_size = self.__read_image(img_path=img_path)
        kps, descs = ([], [])
        with torch.no_grad():
            score_1, coord_1, desc1 = self.keypoint_net(img)
            B, _, Hc, Wc = desc1.shape
            score_1 = torch.cat([coord_1, score_1], dim=1).view(
                3, -1).t().cpu().numpy()
            desc1 = desc1.view(256, Hc, Wc).view(256, -1).t().cpu().numpy()
            desc1 = desc1[score_1[:, 2] > self.conf_threshold, :]
            score_1 = score_1[score_1[:, 2] > self.conf_threshold, :]
            k_best = self.k_best
            if self.k_best < 0:
                k_best = len(score_1)
            kps, descs = select_k_best(score_1, desc1, k_best)
        kps = [cv2.KeyPoint(k[0], k[1], 1.0) for k in kps]
        return kps, descs, img_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='KP2D wrapper',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pretrained_model", type=str,
                        help="pretrained model path")
    parser.add_argument("--image_path", required=True,
                        type=str, help="Input image path")
    parser.add_argument("--conf_threshold", type=float,
                        help="scores confidence threshold", default=0.0)
    parser.add_argument(
        "--k_best", type=int, help="Select k best keypoints based on probability", default=1000)
    args = parser.parse_args()

    img_path = args.image_path
    model_path = args.pretrained_model
    k_best = args.k_best
    conf_threshold = args.conf_threshold

    kp2d_wrapper = Kp2dWrapper(
        model_path=model_path,
        conf_threshold=conf_threshold,
        k_best=k_best
    )

    kpts, desc, size = kp2d_wrapper.process_image(img_path=img_path)
