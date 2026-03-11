"""
test_env.py
============
Quick smoke test to verify the C++ environment is built correctly
and the Gymnasium wrapper works as expected.

Run from parafoil_rl/ after building:
    python test_env.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "python_env"))


def test_cpp_backend():
    """Test the raw C++ backend directly."""
    print("=" * 50)
    print("Test 1: C++ backend (parafoil_cpp)")
    print("=" * 50)

    import parafoil_cpp

    params = parafoil_cpp.SystemParameters()
    print(f"  m_parafoil = {params.m_parafoil} kg")
    print(f"  m_cradle   = {params.m_cradle} kg")

    env = parafoil_cpp.ParafoilEnv(params, 0.0, 0.0, 0.01, 0.1)

    obs = env.reset(seed=42)
    obs_arr = np.array(obs)
    print(f"  obs shape: {obs_arr.shape}  (expected: ({parafoil_cpp.OBS_SIZE},))")
    assert len(obs_arr) == parafoil_cpp.OBS_SIZE, "Wrong obs size"
    assert not np.any(np.isnan(obs_arr)), "NaN in observation"

    action = np.array([0.2, 0.3])
    result = env.step(action)
    obs2, reward, done, info = result
    obs2_arr = np.array(obs2)

    print(f"  After step: reward={reward:.3f}, done={done}")
    print(f"  Info: dist={info['distance_to_target']:.1f}m, alt={info['altitude']:.1f}m")
    assert len(obs2_arr) == parafoil_cpp.OBS_SIZE, "Wrong obs size after step"
    assert not np.isnan(reward), "NaN reward"

    print("  ✓ C++ backend OK")


def test_gym_wrapper():
    """Test the Gymnasium wrapper."""
    print("\n" + "=" * 50)
    print("Test 2: Gymnasium wrapper (ParafoilGymEnv)")
    print("=" * 50)

    from parafoil_gym_env import ParafoilGymEnv

    env = ParafoilGymEnv(target=(0.0, 0.0), dt_action=0.1)

    # Check spaces
    print(f"  obs_space:  {env.observation_space}")
    print(f"  act_space:  {env.action_space}")

    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, "Wrong obs shape"
    assert obs.dtype == np.float32, "Wrong obs dtype"
    print(f"  reset obs: {obs}")

    # Check obs is within bounds
    assert np.all(obs >= env.observation_space.low - 0.01), "Obs below lower bound"
    assert np.all(obs <= env.observation_space.high + 0.01), "Obs above upper bound"

    # Step with zero action
    action = np.zeros(2, dtype=np.float32)
    obs2, reward, terminated, truncated, info = env.step(action)
    print(f"  step: reward={reward:.3f}, term={terminated}, trunc={truncated}")

    # Run 50 steps
    total_reward = reward
    for _ in range(49):
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"  50 steps: total_reward={total_reward:.1f}")
    print("  ✓ Gymnasium wrapper OK")


def test_episode():
    """Run a full episode to ground."""
    print("\n" + "=" * 50)
    print("Test 3: Full episode (random policy)")
    print("=" * 50)

    from parafoil_gym_env import ParafoilGymEnv

    env = ParafoilGymEnv(target=(0.0, 0.0), dt_action=0.1, max_episode_time=600.0)
    obs, info = env.reset(seed=7)

    total_reward = 0.0
    n_steps = 0

    while True:
        action = np.array([0.0, 0.0], dtype=np.float32)  # Symmetric zero brake
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        n_steps += 1

        if terminated or truncated:
            break

        if n_steps % 200 == 0:
            state = env.get_state()
            print(f"  t={env.get_time():.0f}s  alt={-state[2]:.0f}m  "
                  f"dist={info['distance_to_target']:.0f}m")

    print(f"  Episode done: steps={n_steps}, reward={total_reward:.1f}")
    print(f"  Landing error: {info['distance_to_target']:.1f} m")
    print(f"  Hit ground: {info['hit_ground']}, Diverged: {info['diverged']}")
    print("  ✓ Full episode OK")


if __name__ == "__main__":
    try:
        test_cpp_backend()
    except ImportError as e:
        print(f"\n  ✗ parafoil_cpp not found: {e}")
        print("  Build the module first: see README.md")
        sys.exit(1)

    try:
        test_gym_wrapper()
        test_episode()
    except Exception as e:
        print(f"\n  ✗ Test failed: {e}")
        raise

    print("\n" + "=" * 50)
    print("  All tests passed! Ready to train.")
    print("  Run: cd training && python train_ppo.py --n_envs 1 --total_steps 50000")
    print("=" * 50)
