"""Predict one binary mask with a best.pt checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['args']
    model = build_model(config['model'], base_channels=config['base_channels']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    with Image.open(args.image) as source:
        image = source.convert('RGB')
    width, height = image.size
    size = config['image_size']
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))
        probability = F.interpolate(probability, size=(height, width), mode='bilinear', align_corners=False)
        mask = (probability[0, 0].cpu().numpy() >= 0.5).astype('uint8') * 255
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output)
    print(output)


if __name__ == '__main__':
    main()
