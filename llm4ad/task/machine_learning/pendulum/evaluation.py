from __future__ import annotations

from typing import Any
import gym
import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.machine_learning.pendulum.template import template_program, task_description

__all__ = ['Pendulum']


def evaluate(env: gym.Env, action_select: callable) -> float | None:
    try:
        fitness = []
        # Parallel evaluation 4 times, core=4
        # fitness = Parallel(n_jobs=4)(delayed(evaluate_single)(env, action_select) for _ in range(5))
        for i in range(5):
            fitness.append(evaluate_single(env, action_select))
        fitness = np.mean(fitness)

        return fitness
    except Exception as e:
        return None


def evaluate_single(env: gym.Env, action_select: callable) -> float:
    """Evaluate heuristic function on the pendulum swing-up problem."""

    observation, _ = env.reset()  # initialization
    action = 0.0  # initial action (torque)
    total_reward = 0

    for i in range(env._max_episode_steps + 1):  # protect upper limits
        action = action_select(observation[0],  # cos(theta)
                               observation[1],  # sin(theta)
                               observation[2],  # angular velocity
                               action)  # last action (torque)
        observation, reward, done, truncated, info = env.step([action])
        total_reward += reward

        if done or truncated:
            # self.env.close()
            cos_theta = observation[0]
            sin_theta = observation[1]
            angular_velocity = observation[2]

            # Calculate error terms
            angle_error = abs(1 - cos_theta)  # Distance from vertical (cos(theta) = 1 when upright)
            stability_error = abs(sin_theta)  # Penalize instability

            # Total error
            error = angle_error + stability_error

            # Fitness calculation: ensure fitness > 1 and closer to 1 for better states
            fitness = 1 + error
            if fitness <= 1:
                return -(i + 1) / env._max_episode_steps
            else:
                return -fitness


class Pendulum(Evaluation):
    """Evaluator for the pendulum swing-up problem."""

    def __init__(self, max_steps=500, **kwargs):
        """
            Args:
                - 'max_steps' (int): Maximum number of steps allowed per episode in the Pendulum-v1 environment (default is 200).
                - '**kwargs' (dict): Additional keyword arguments passed to the parent class initializer.

            Attributes:
                - 'env' (gym.Env): The Pendulum-v1 environment with a modified maximum episode length.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=20
        )

        self.env = None
        self.env = gym.make('Pendulum-v1')
        self.env._max_episode_steps = max_steps

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return evaluate(self.env, callable_func)
