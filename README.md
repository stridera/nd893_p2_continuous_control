[//]: # "Image References"
[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Udacity Deep Reinforcement Learning Nanodegree

## Project 1: Navigation

This is the second project in the Udacity [Deep Reinforcement Learning Nanodegree.](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) The goal is to setup a deep RL model that will play the supplied [Banana Environment](env.md).

### Getting Started

1. Checkout this repo.

2. Download the required environment. (This is the 20 agent version.)

   - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
   - [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
   - [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
   - [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

3. Unzip the file in a known location. The default location for the script is in `./env/` but you can provide your own path to the script.

4. Create and configure your virtual environment:

   1. Create and activate a new virtual environment.

      ```bash
      python3 -mvenv .venv
      source .venv/bin/activate
      ```

   2. Install dependencies:

      ```bash
      pip install python/
      ```

   3. (Optional) Train a new model. Warning: This will overwrite the supplied models. Note: Add `--help` to the command below to see the options and defaults.

      ```bash
      python train.py # Run with default options
      python train.py --env reacher_env/  --episodes 1000  --seed 0 # Run with given arguments.
      ```

   4. View existing models. Note: This command also supports the `--help` parameter to see options and defaults.

      ```bash
      python play.py # Runs the final.pth model with default params.
      python play.py --env reacher_env/ --model_path models/solved --fps 10 # Run the solved model at 10 frames per second.
      ```

Side Quest:
If you want to try using Stable Baselines3 to train and run the model, there is an additional script for you. Complete step
1 and 2 from above and then:

```bash
python3 ​train_sb3.py --train # Train a new model
python3 ​train_sb3.py # Evaluate the trained model
```

### Environment Details

For this project, you are required to keep your arm inside the blue orb.

![Trained Agent][image1]
Your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
- This yields an **average score** for each episode (where the average is over all 20 agents).

### Trained Models

There are two models included. A final and solved model. The solved model was saved as soon as the model achieved the task of getting a score of +13 over 100 consecutive episodes. This happened after around 500 episodes. The final episode was trained until the curve flattened out, somewhere around 600 episodes with an average score around 16.

More information can be found in the [training report](Report.md).
