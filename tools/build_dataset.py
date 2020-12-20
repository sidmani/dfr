from pathlib import Path
from argparse import ArgumentParser
from dfr.ckpt import Checkpoint
from PIL import Image
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--data',
        '-d',
        dest='data',
        required=True,
        help='The source directory for the 3D-R2N2 shapenet rendering dataset'
    )
    parser.add_argument(
        '-o',
        dest='output',
        required=True,
        help='The output directory'
    )
    parser.add_argument(
        '-r'
        '--resize',
        dest='resize',
        default=None,
        help='Resize to a new resolution'
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(exist_ok=True)
    dataPath = Path(args.data)

    objects = list(dataPath.glob('*'))
    imgsPerFolder = 24

    for folder in tqdm(objects):
        idx = np.random.randint(0, imgsPerFolder)
        imgPath = Path(folder / 'rendering' / f'{idx:02d}.png')
        dest = out / f'{folder.name}.png'

        img = Image.open(imgPath)
        if args.resize is not None:
            s = int(args.resize)
            img = img.resize((s, s))
        img.save(dest)
