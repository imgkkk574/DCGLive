# DCGLive: A Dynamic Collaboration-Aware Graph Learning Approach for Live Streaming Recommendation

This repository contains the implementation for the paper: "Room Matters: Dynamic Room-level Collaboration Information Modeling for Live Streaming Recommendation."

## How to Run the Code

### Train DCGLive

1. Create a directory to save the models:
    ``` mkdir saved_models ```
2. Train the DCGLive model:
    ``` python DCGLive.py --network kuailive --model DCGLive --epochs 50 --method attention --embedding_dim 128 --alpha_LiveCI 0.9 ```

### Evaluate DCGLive

To evaluate the DCGLive model, you can use the following command:
``` python evaluate_interaction_prediction.py --network kuailive --model DCGLive --method attention --epoch $epoch --embedding_dim 128 --alpha_LiveCI 0.9 ```

Alternatively, you can run the script `run_eva.sh` to evaluate all epochs.

## More Details

More details about experiments and datasets can be found at ``` More Details about Experiments and Datasets.pdf ``` in our repo.