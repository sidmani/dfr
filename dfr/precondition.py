import torch
from torch.optim import Adam
from tqdm import tqdm

# precondition the SDF network to represent a sphere
def precondition(ckpt, device, radius=0.4, steps=5000, batch=128, lr=4e-5, logger=None):
    opt = Adam(ckpt.gen.parameters(), lr)
    for i in tqdm(range(steps)):
        opt.zero_grad(set_to_none=True)
        pts = torch.randn((batch, 3), device=device, requires_grad=True)
        z = torch.normal(0.0, ckpt.hparams.latentStd, (batch, ckpt.hparams.latentSize), device=device)
        truth = pts.norm(dim=1, keepdim=True) - radius

        out = ckpt.gen(pts, z, mask=torch.ones((batch,), device=device, dtype=torch.bool), geomOnly=True)
        grad = torch.autograd.grad(outputs=out,
                inputs=pts,
                grad_outputs=torch.ones_like(out),
                create_graph=True)[0]

        eikonal = ((grad.norm(dim=1) - 1.) ** 2.).mean()
        loss = ((out - truth) ** 2).mean() + 5 * eikonal
        loss.backward()
        opt.step()
        if logger is not None:
            logger.logger.add_scalar('data/precondition', loss.detach(), global_step=i)

    if loss.item() > 0.5:
        raise Exception('Preconditioning failed; bad initialization.')
