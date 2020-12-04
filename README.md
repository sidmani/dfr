# Renderer
The renderer uses multiple passes in increasing resolution to approximately figure out where the object is, with autograd disabled, and does a final evaluation on the critical points with autograd on. The speed of the renderer is bound to the total number of SDF queries, and the memory usage is tied to the number of queries in the final pass (due to autograd). It's possible to decrease the mean memory usage by basing the density of the final evaluation on the resolution used at a certain pixel, but this introduces some grid artifacts. Also, decreasing the mean is unhelpful, since the max usage determines the batch size. This algorithm is available on the branch `memory`.

The need for fuzzing the edges to help geometry learning likely degrades the quality of the edges. Need to investigate other methods of propagating gradients through free space.

During training, grid artifacts appear around epoch 3-5k and disappear around epoch 12-14k. These are likely caused by the periodic positional encoding's effect on the texture network, and not due to the ray grid. Jittering the rays does not reduce these artifacts and should not be used; the jittering introduces errors in the multiscale refinement process and degrades the quality of edges. However, it is important to randomize the starting distance of rays in order to provide supervision to more points on the surface- otherwise, the sampled points all lie on concentric circles.

I tried taking the mean over various points within a pixel in order to reduce grid artifacts, but this increases memory usage 3-4x and doesn't seem to improve anything. 

The renderer imposes an approximation to the unit circle on the output image at the minimum used resolution. This can cause problems at certain view angles for long objects, and should be fixed. **TODO**

Ultimately, I suspect that it is impossible to get usable results at less than 128x128 resolution, but at that resolution only 12 objects can fit in 16GB of VRAM (again limited by the autograd memory usage). An easy solution is to use the 32GB V100, which may take around 40 hours to train 100k epochs with a batch size of 32.

# Generator
It is likely that the Eikonal gradient penalty causes the frequency of the textures to decrease, since the textures currently have only 2 separate layers and depend on the backbone network for the other 6. **TODO**

SIRENs may improve the quality of the generated output. (Does this conflict with positional encoding?) Also, like SALD, softplus may be useful.

The positional encoding can be increased to 10 frequencies as in NeRF, but this is unlikely to help much.

Weight norm doesn't seem to change much, so I'm keeping it enabled in case it improves the converged solution at all.

GRAF uses a separate latent code for the texture, which could be very useful. Try this after stuff works better.

A deeper branch architecture (4 layers) may improve textures a lot, because the skip connection can go directly into the texture network.

A different colorspace (like HSV) may be useful because the lightening/darkening of a surface can be conducted via color or surface normal, and I think it's biased towards changing the normal instead of darkening the color, casuing dents where dark spots should be.

# Discriminator
It has been suggested that the discriminator cannot handle multiple views of the object, but in practice I don't think this is an issue.

Looks like the standard GAN loss (based on the KL divergence) is unstable with this architecture. Using the WGAN loss without regularization causes (as expected) exploding gradients and violates the Lipschitz condition on the discriminator. I've tried 3 methods for enforcing the Lipschitz constraint:

- *WGAN-GP*
This is the method recommended in DFR, and it works better than any other method. One of the concerns is that it's computationally expensive, but the running the generator is so expensive that the gradient penalty is a small fraction of total time. It may have a large memory impact, but this needs investigation.

- *Spectral Norm*
Spectral norm doesn't work with the standard GAN loss (unstable training, mode collapse, vanishing gradient), but stabilizes training with the WGAN loss. However, the results are incredibly degraded and unusable. It seems like this is a known problem; spectral norm doesn't work with WGAN loss even though it was designed to do so.

- *Sphere-GAN + WGAN loss*
The gradients vanish rapidly and it's impossible to get any results. I tried more than a hundred different hyperparameter settings and architectures tweaks.

The eikonal factor is moderately important, because a high factor may significantly hamper texture learning and fine geometric features. The model still learns a renderable object with the factor set to 0, but it's prone to having blobs appear far from the object along with other artifacts.

It's fairly clear that the discriminator is powerful enough for the current setting, but more experimentation is needed with different discriminator/generator update ratios.

# Training 
The generator's gradients aren't used in the iterations where it's not updated, but they're required for computing normals, so we can't use `no_grad` here.

AMP can significantly reduce memory usage and is a high-priority task. **TODO**

There's some sort of glitch which causes multiple tensorboard event files to be created when opening the checkpoint (e.g. to view a mesh).

The downsampling for the training data causes bad pixels to appear, try a better interpolation algorithm. 
