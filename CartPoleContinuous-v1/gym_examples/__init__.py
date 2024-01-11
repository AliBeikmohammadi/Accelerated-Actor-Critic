from gym.envs.registration import register

register(
    id="gym_examples/CartPoleContinuous-v1",
    entry_point="gym_examples.envs:CartPoleContinuous_v1",
    max_episode_steps=500,
    reward_threshold=475.0,
)
