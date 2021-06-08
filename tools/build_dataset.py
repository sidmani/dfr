from pathlib import Path
from argparse import ArgumentParser
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
        help='The source directory for the images'
    )
    parser.add_argument(
        '-o',
        dest='output',
        required=True,
        help='The output directory'
    )
    parser.add_argument(
        '--random-count',
        dest='count',
        default=1,
        help='# of views to select (max 24)',
        type=int,
    )
    # parser.add_argument(
    #     '--resize',
    #     '-r',
    #     dest='resize',
    #     default=None,
    #     help='Resize to a new resolution',
    #     type=int,
    # )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(exist_ok=True)
    dataPath = Path(args.data)

    objects = list(dataPath.glob('*'))
    imgsPerFolder = 24

    rng = np.random.default_rng()

    for folder in tqdm(objects):
        idxs = rng.choice(imgsPerFolder, size=args.count)
        for i in range(args.count):
            imgPath = Path(folder / 'rendering' / f'{idxs[i]:02d}.png')
            dest = out / f'{folder.name}_{idxs[i]}.png'

            img = Image.open(imgPath)
            # if args.resize is not None:
            #     img = img.resize((args.resize, args.resize))
            img.save(dest)
