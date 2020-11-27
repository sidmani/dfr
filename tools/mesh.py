import numpy as np
import torch
from argparse import ArgumentParser
from skimage import measure
from trimesh import Trimesh, Scene
from trimesh.viewer.windowed import SceneViewer
from pathlib import Path
from dfr.checkpoint import Checkpoint

def main(args):
    ckpt = Checkpoint(Path.cwd() / 'runs',
                      version=args.ckpt,
                      epoch=args.epoch,
                      device=None)

    res = int(args.res)

    with torch.no_grad():
        points = torch.linspace(-1.0, 1.0, res)
        x, y, z = torch.meshgrid(points, points, points)

        # grid has dimension res^3 x 3
        grid = torch.cat([
                torch.flatten(x).unsqueeze(1),
                torch.flatten(y).unsqueeze(1),
                torch.flatten(z).unsqueeze(1),
            ], dim=1)

        latent = torch.normal(
                mean=0.0,
                std=ckpt.hparams.latentStd,
                size=(1, ckpt.hparams.latentSize))

        expandedLatents = latent.expand(grid.shape[0], -1)

        # create input vector and compute values
        # out, normals = gen.sdf(grid, latent)
        out = ckpt.gen.sdf(grid, expandedLatents, geomOnly=True)
        # reshape and return a 3D grid
        # TODO: does this cause rotation?
        cubic = torch.reshape(out, (res, res, res)).detach().numpy()

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
            dest='ckpt',
            required=True,
    )
    parser.add_argument(
            '--resolution',
            '-r',
            dest='res',
            default=64,
    )

    args = parser.parse_args()
    main(args)
