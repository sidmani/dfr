The network becomes unstable and prone to forgetting when the progressive growing size increases.
Possible causes:
1. Difference between full-size real data used in previous stage and half-sized data in current stage
This is fully prevented by reuse of identical real data. Not the cause.

2. Difference between half size raycasting and full size + downsampled version
There was a bug in grid alignment. With that fixed, the switch in resolution does not cause any problems.

3. float16 underflow
Not the problem- issue still occurs without AMP. However, the gradient scaler should be started at 2048 instead of 32768.

TODO: understand how the equalized learning rate works and why it solves the problem / introduces different instability.implement it manually.

See if any other factors are relevant (eikonal, disc iterations)
