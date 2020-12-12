#!/bin/bash
nohup tensorboard --logdir=runs --samples_per_plugin images=1000 > /dev/null &
