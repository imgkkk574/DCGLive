#!/bin/bash

for epoch in {0..49}
do
    python evaluate_interaction_prediction.py --network kuailive --model DCGLive --method attention --epoch $epoch --embedding_dim 128 --alpha_LiveCI 0.9
done