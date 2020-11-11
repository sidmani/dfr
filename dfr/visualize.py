import numpy as np
import torch
from argparse import ArgumentParser
from skimage import measure
from .gan import GAN
from trimesh import Trimesh, Scene
from trimesh.viewer.windowed import SceneViewer
from pathlib import Path
import re

def main(args):
    if args.epoch is None:
        epochs = Path(f"lightning_logs/version_{args.version}/checkpoints/").glob('*.ckpt')
        epochs = list(map(lambda x: int(re.match(r"epoch=([0-9]+)", x.stem).group(1)), epochs))
        epoch = max(epochs)
        print(f"Loading epoch {epoch}")
    else:
        epoch = args.epoch

    res = int(args.res)


    model = GAN.load_from_checkpoint(f"lightning_logs/version_{args.version}/checkpoints/epoch={epoch}.ckpt")

    if args.object is not None:
        objId = int(args.object)
    else:
        objId = np.random.randint(0, model.embedding.num_embeddings)

    points = torch.linspace(-0.55, 0.55, res)
    x, y, z = torch.meshgrid(points, points, points)

    # grid has dimension res^3 x 3
    grid = torch.cat([
            torch.flatten(x).unsqueeze(1),
            torch.flatten(y).unsqueeze(1),
            torch.flatten(z).unsqueeze(1),
        ], dim=1)

    latent = torch.rand(1, model.hparams.latentSize)

    # create input vector and compute values
    out = model.gen.sdf(grid, latent)
    # reshape and return a 3D grid
    # TODO: does this cause rotation?
    cubic = torch.reshape(out, (res, res, res))

    verts, faces, normals, values = measure.marching_cubes(cubic, 0)
    mesh = Trimesh(vertices=verts, faces=faces)
    scene = Scene([mesh])
    viewer = SceneViewer(scene)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
            '--epoch',
            '-e',
            dest='epoch',
            default=None,
    )
    parser.add_argument(
            '--version',
            '-v',
            dest='version',
            default=None,
    )
    parser.add_argument(
            '--object',
            '-o',
            dest='object',
            default=None,
    )
    parser.add_argument(
            '--resolution',
            '-r',
            dest='res',
            default=64,
    )

    args = parser.parse_args()
    main(args)
