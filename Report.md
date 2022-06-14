[//]: # "Image References"
[scores]: images/graph.png "Training Score Graph"

# Training Report

## Deep Deterministic Policy Gradient (DDPG)

DDPG is a off-policy model designed for continuous actions. (Actions that aren't simply up, down, left right, but is
more like 0.2 to the left, and 0.8 up.)

### Model Architecture

The network followed the paper with 3 linear layers linked with ReLU (Rectified Linear Unit) activation functions.

The model input and output were obtained directly from the environment.
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry is clipped between
-1 and 1.

So the final trained models network looks like this:

```python
Actor(
  (fc1): Linear(in_features=33, out_features=400, bias=True)
  (bn1): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=400, out_features=300, bias=True)
  (fc3): Linear(in_features=300, out_features=4, bias=True)
)
Critic(
  (fcs1): Linear(in_features=33, out_features=400, bias=True)
  (bn1): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=404, out_features=300, bias=True)
  (fc3): Linear(in_features=300, out_features=1, bias=True)
)
```

The Actor returns 4 actions while the critic returns a boolean determining if it thinks the action is a good move or not.

We use an Adam optimizer to train the models and a mean square function for determining loss.

Other noticable actions we do include adding noise to the action to give some randomness in the training.

### Hyper-parameters

The model was trained using the following hyper-parameters:

- Buffer Size: `1,000,000` We keep up to 1M samples around to train the model with.
- Batch Size: `128` Choosing a batch size is difficult. 128 seems like a good place for memory efficiency and training speed.
- Gamma: `0.99` The gamma factor is used to determine the reward discount. This will slowly discount the reward to make it so the chain of actions leading toward a positive reward are recognized.
- Tau: `0.001` The tau value is used to soft update the target network. This means we only slowly update the target network using the following update schedule: *θ*−=*θ*×*τ*+*θ*−×(1−*τ*)
- LR_ACTOR: `0.0004` - The Actor learning rate.
- LR_CRITIC: `0.004` - The Critic Learning Rate. Played around with a bunch of values and this worked well. Should be kept
  higher than or equal to the actor learning rate.

## Results

The model trained quickly, reaching a solved status (+30/100eps) at episode 116. When we allowed the model to continue, it would generally improve slightly more and peak at around +38/100 eps. Interestingly, when we train using PPO (via Stable Baselines3)
it takes significantly longer.

![Scores Graph][scores]
DDPG: Orange PPO: Blue

## Future Improvements

DDPG worked really well for this environment. I had to separate the training step from the action step (and force the
call to happen manually) to link the trainings with the episode step. It appeared to not train if I just tied it to learn
every x timesteps.

I'm really excited to try this with the crawler model. Using Stable Baselines3 to run them allowes me to run quick tests
using various professional trained models. I'm interesting to see how the various models learn and play the environment.
