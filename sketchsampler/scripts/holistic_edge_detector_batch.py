# script needs some changes
# raw source script from https://github.com/sniklaus/pytorch-hed
# need to change loading & processing if going down this approach


import os
import cv2
import getopt
import numpy
import PIL
import PIL.Image
import sys
import torch
import pathlib
import argparse


##########################################################
torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################
IMG_DIR = "/home/srk/Downloads/pytorch-hed-master/images"
OUT_DIR = "/home/srk/Downloads/pytorch-hed-master/out_images"
DO_THRESHOLD = True
THRESHOLD = 90  # needs more tweaking?

##########################################################
# DONT CHANGE THESE NIV
base_width = 480
height = 320
arguments_strModel = 'bsds500' # only 'bsds500' for now
##########################################################


def parse_args():

    parser = argparse.ArgumentParser('Image2LineDrawing', add_help=False)

    parser.add_argument('--img_dir', type=str, default=IMG_DIR)
    parser.add_argument('--out_dir', type=str, default=OUT_DIR)
    parser.add_argument('--threshold', type=int, default=THRESHOLD)
    parser.add_argument('--do_threshold', action='store_true', default=DO_THRESHOLD)

    return parser.parse_args()



# arguments_strIn = './images/sample.png'
# arguments_strOut = './out.png'

# for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
#     if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use
#     if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
#     if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-hed/network-' + arguments_strModel + '.pytorch', file_name='hed-' + arguments_strModel).items() })
    # end

    def forward(self, tenInput):
        tenInput = tenInput * 255.0
        tenInput = tenInput - torch.tensor(data=[104.00698793, 116.66876762, 122.67891434], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))
    # end
# end

netNetwork = None

##########################################################

def estimate(tenInput):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().eval()
    # end

    intWidth = tenInput.shape[2]
    intHeight = tenInput.shape[1]

    assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    return netNetwork(tenInput.cuda().view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()
# end

##########################################################

if __name__ == '__main__':
    args = parse_args()


    img_paths = sorted(list(pathlib.Path(args.img_dir).glob('*.png')))

    for img_path in img_paths:

        # tenInput = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(img_path))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
        #                 1.0 / 255.0)))

        _cvi = cv2.imread(str(img_path))
        if _cvi.shape[0] != 320:
            _cvi = cv2.resize(_cvi, (base_width, height), interpolation = cv2.INTER_AREA)
            cv2.imwrite(str(img_path), _cvi)

        tenInput2 = PIL.Image.open(img_path).resize((base_width, height), PIL.Image.Resampling.LANCZOS)
        tenInput2 = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(tenInput2)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
                1.0 / 255.0)))

        tenOutput = estimate(tenInput2)

        tenOutput = PIL.Image.fromarray(
            (tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8))
        tenOutput = tenOutput.resize((256, 256), PIL.Image.Resampling.LANCZOS)
        if args.do_threshold:
            tenOutput = tenOutput.point(lambda p: p > args.threshold and 255)
        tenOutput.save(os.path.join(args.out_dir, img_path.name))
    # end
