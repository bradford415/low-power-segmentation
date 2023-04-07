from argparse import ArgumentParser, Namespace
from typing import List, Tuple, BinaryIO

import cv2
import numpy
import torch
import torch.nn.functional as F
from cv2.mat_wrapper import Mat
from imageio.core.util import Array
from imageio import imread
from numpy import ndarray
from torch import Tensor
from torchvision.transforms import transforms
from networks import deeplabv3
from utils.fanet import FANet

import pkg_resources
import os

SIZE: List[int] = [512, 512]


def getArgs() -> Namespace:
    # NOTE: These variables can be changed
    programName: str = "LPCVC 2023 Sample Solution"
    authors: List[str] = ["Nicholas M. Synovic", "Ping Hu"]

    prog: str = programName
    usage: str = f"This is the {programName}"
    description: str = f"This {programName} does create a single segmentation map of arieal scenes of disaster environments captured by unmanned arieal vehicles (UAVs)"
    epilog: str = f"This {programName} was created by {''.join(authors)}"

    # NOTE: Do not change these flags
    parser: ArgumentParser = ArgumentParser(prog, usage, description, epilog)
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Filepath to an image to create a segmentation map of",
    )
    parser.add_argument( 
        "-o",
        "--output",
        required=True,
        help="Filepath to the corresponding output segmentation map",
    )

    return parser.parse_args()

def loadImageToTensor(imagePath: str) -> Tensor:
    MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    STANDARD_DEVIATION: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    print(imagePath)

    image: Array = imread(uri=imagePath)
    resizedImage: Mat = cv2.resize(image, tuple(SIZE), interpolation=cv2.INTER_AREA)
    imageTensor: Tensor = transforms.ToTensor()(resizedImage)
    imageTensor: Tensor = transforms.Normalize(mean=MEAN, std=STANDARD_DEVIATION)(
        imageTensor
    )
    imageTensor: Tensor = imageTensor.unsqueeze(0)

    return imageTensor


def main() -> None:

    args: Namespace = getArgs()

    model_weights: BinaryIO = pkg_resources.resource_stream(__name__, "model.pt")


#    model_weights="model.pt"

    device = torch.device('cpu')

    imageTensor: Tensor = loadImageToTensor(imagePath=args.input)
    
    model = getattr(deeplabv3, 'create_resnet101')(
            pretrained=False,
            device=device,
            num_classes=14
            )
            
    model = model.eval()
    
    weights = torch.load(model_weights, map_location=device)
    
    state_dict = {k[7:]: v for k,
                v in weights['model'].items()}

    model.load_state_dict(state_dict)


    outTensor: Tensor = model(imageTensor)
    outTensor: Tensor = F.interpolate(
        outTensor, SIZE, mode="bilinear", align_corners=True
    )

    outArray: ndarray = outTensor.cpu().data.max(1)[1].numpy()
    outArray: ndarray = outArray.astype(numpy.uint8)

    outImage: ndarray = cv2.cvtColor(outArray[0], cv2.COLOR_GRAY2BGR)
    cv2.imwrite(args.output, outImage)
