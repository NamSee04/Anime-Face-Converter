import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from src.model import net
from src.model.loss import adaptive_instance_normalization, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0, interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def main():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', type=str, default='/home/namsee/Desktop/SCHOOL/cs431/StyleTransfer/src/data/face2anime/test/testA',
                    help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str, default='/home/namsee/Desktop/SCHOOL/cs431/StyleTransfer/src/data/face2anime/test/testB',
                    help='Directory path to a batch of style images')
    parser.add_argument('--vgg', type=str, default='./src/ckpt/vgg_normalised.pth')
    parser.add_argument('--decoder', type=str, default='/home/namsee/Desktop/SCHOOL/cs431/StyleTransfer/src/ckpt/decoder.pth')

    # Additional options
    parser.add_argument('--content_size', type=int, default=256,
                        help='New (minimum) size for the content image, '
                             'keeping the original size if set to 0')
    parser.add_argument('--style_size', type=int, default=256,
                        help='New (minimum) size for the style image, '
                             'keeping the original size if set to 0')
    parser.add_argument('--crop', action='store_true',
                        help='Do center crop to create squared image')
    parser.add_argument('--save_ext', default='.jpg',
                        help='The extension name of the output image')
    parser.add_argument('--output', type=str, default='output4fid',
                        help='Directory to save the output image(s)')

    # Advanced options
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='The weight that controls the degree of '
                             'stylization. Should be between 0 and 1')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #dir
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]
    print(len(content_paths), len(style_paths))

    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    content_tf = test_transform(args.content_size, args.crop)
    style_tf = test_transform(args.style_size, args.crop)

    for i in range(len(content_paths)):
        content_path = content_paths[i]
        style_path = style_paths[i]
        content = content_tf(Image.open(str(content_path)))
        style = style_tf(Image.open(str(style_path)))
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha)
        output = output.cpu()
        output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
        save_image(output, str(output_name))

if __name__ == '__main__':
    main()

# python -m pytorch_fid /home/namsee/Desktop/SCHOOL/cs431/StyleTransfer/output4fid  /home/namsee/Desktop/SCHOOL/cs431/StyleTransfer/src/data/face2anime/test/testB --device cuda:1
# z20 0.6 100