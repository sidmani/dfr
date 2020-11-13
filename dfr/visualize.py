import numpy as np
import torch
from argparse import ArgumentParser
from skimage import measure
from trimesh import Trimesh, Scene
from trimesh.viewer.windowed import SceneViewer
from pathlib import Path
from .generator import Generator
from .sdfNetwork import SDFNetwork
from .raycast.frustum import Frustum
import re

def main(args):
    if args.epoch:
        epoch = int(args.epoch)
    else:
        # load the newest checkpoint for given version
        checkpointPath = Path.cwd() / 'runs' / f"v{args.ckpt}"
        if not checkpointPath.exists:
            raise Exception(f'Version {args.ckpt} does not exist')

        available = list(checkpointPath.glob('*.pt'))
        if len(available) == 0:
            raise Exception(f'No checkpoints found for version {args.ckpt}')

        nums = []
        for f in available:
            match = re.match("e([0-9]+)", str(f.stem))
            nums.append(int(match[1]))
        epoch = max(nums)

    checkpoint = torch.load(checkpointPath / f"e{epoch}.pt", map_location=torch.device('cpu'))
    version = int(args.ckpt)
    print(f"Loaded version {version}, epoch {checkpoint['epoch']}.")

    hp = checkpoint['hparams']
    frustum = Frustum(2 * np.pi / 3, 128, device=None)
    sdf = SDFNetwork(hp)
    gen = Generator(sdf, frustum, hp)
    gen.load_state_dict(checkpoint['gen'])
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
                std=1e-2,
                size=(1, hp.latentSize))

        # create input vector and compute values
        out, normals = gen.sdf(grid, latent)
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
