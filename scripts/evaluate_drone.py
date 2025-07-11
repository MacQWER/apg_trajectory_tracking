import os
import time
import argparse
import json
import numpy as np
import torch

from neural_control.environments.drone_env import QuadRotorEnvBase
from neural_control.environments.rendering import animate_quad
from neural_control.trajectory.straight import Hover, Straight
from neural_control.trajectory.circle import Circle
from neural_control.trajectory.polynomial import Polynomial
from neural_control.trajectory.random_traj import Random
from neural_control.dataset import QuadDataset
from neural_control.controllers.network_wrapper import NetworkWrapper
from neural_control.controllers.mpc import MPC
from neural_control.dynamics.quad_dynamics_flightmare import FlightmareDynamics
from neural_control.dynamics.quad_dynamics_simple import SimpleDynamics
from evaluate_base import run_mpc_analysis, load_model_params, average_action
try:
    from neural_control.flightmare import FlightmareWrapper
except ModuleNotFoundError:
    pass

ROLL_OUT = 1

# Use cuda if available
# device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuadEvaluator():

    def __init__(
        self,
        controller,
        environment,
        ref_length=5,
        max_drone_dist=0.1,
        render=0,
        dt=0.05,
        test_time=0,
        speed_factor=.6,
        train_mode="concurrent",
        **kwargs
    ):
        self.controller = controller
        self.eval_env = environment
        self.horizon = ref_length
        self.max_drone_dist = max_drone_dist
        self.render = render
        self.dt = dt
        self.action_counter = 0
        self.test_time = test_time
        self.speed_factor = speed_factor
        self.train_mode = train_mode
        if hasattr(self.controller.net, "reset_hidden_state"):
            # if it's an lstm based model, reset the hidden state
            self.controller.net.reset_hidden_state()

    def help_render(self, t_prev):
        """
        Helper function to make rendering prettier
        """
        if self.render:
            # print([round(s, 2) for s in current_np_state])
            current_np_state = self.eval_env._state.as_np
            self.eval_env._state.set_position(
                current_np_state[:3] + np.array([0, 0, 1])
            )
            self.eval_env.render()
            self.eval_env._state.set_position(current_np_state[:3])

            # sleep to make the simulation realistic
            time_now = time.time()
            dt_process = (time_now - t_prev)
            dt_sleep = max(0.0, self.dt - dt_process)
            time.sleep(dt_sleep)
            t_prev = time_now + dt_sleep
        return t_prev

    def follow_trajectory(
        self,
        traj_type,
        max_nr_steps=200,
        thresh_stable=.4,
        thresh_div=3,
        do_avg_act=0,
        **traj_args
    ):
        """
        Follow a trajectory with the drone environment
        Argument trajectory: Can be any of
                straight
                circle
                hover
                poly
        """
        # reset action counter for new trajectory
        self.action_counter = 0

        # reset drone state
        init_state = [0, 0, 3]
        self.eval_env.zero_reset(*tuple(init_state))

        states = None  # np.load("id_5.npy")
        # Option to load data
        if states is not None:
            self.eval_env._state.from_np(states[0])

        # get current state
        current_np_state = self.eval_env._state.as_np

        # Get right trajectory object:
        object_dict = {
            "hover": Hover,
            "straight": Straight,
            "circle": Circle,
            "poly": Polynomial,
            "rand": Random
        }
        reference = object_dict[traj_type](
            current_np_state.copy(),
            self.render,
            self.eval_env.renderer,
            max_drone_dist=self.max_drone_dist,
            speed_factor=self.speed_factor,
            horizon=self.horizon,
            dt=self.dt,
            test_time=self.test_time,
            **traj_args
        )
        if traj_type == "rand":
            # self.eval_env._state.from_np(reference.initial_state)
            current_np_state = self.eval_env.zero_reset(
                *tuple(reference.initial_pos)
            )

        t_prev = self.help_render(time.time())

        (reference_trajectory, drone_trajectory, divergences,
         actions) = [], [current_np_state], [], []
        for i in range(max_nr_steps):
            # acc = self.eval_env.get_acceleration()
            trajectory = reference.get_ref_traj(current_np_state, 0)
            action = self.controller.predict_actions(
                current_np_state, trajectory.copy()
            )

            # possible average with previous actions
            # use_action = average_action(action, i, do_avg_act=do_avg_act)
            if self.train_mode == "concurrent":
                use_action = action[0]
            else:
                use_action = action
            actions.append(action)

            current_np_state, stable = self.eval_env.step(
                use_action, thresh=thresh_stable
            )
            # np.set_printoptions(suppress=1, precision=3)
            # print(current_np_state[:3], trajectory[0, :3])
            if states is not None:
                self.eval_env._state.from_np(states[i])
                current_np_state = states[i]

            t_prev = self.help_render(t_prev)

            drone_pos = current_np_state[:3]
            drone_trajectory.append(current_np_state)

            # project to trajectory and check divergence
            drone_on_line = reference.project_on_ref(drone_pos)
            reference_trajectory.append(drone_on_line)
            div = np.linalg.norm(drone_on_line - drone_pos)
            divergences.append(div)

            # reset the state to the reference
            if div > thresh_div or not stable:
                if self.test_time:
                    # TODO: must always be down for flightmare train
                    # print("diverged at", len(drone_trajectory))
                    break
                current_np_state = reference.get_current_full_state()
                self.eval_env._state.from_np(current_np_state)

            if i >= reference.ref_len:
                break
        if self.render:
            self.eval_env.close()
        # return trajectorie and divergences
        return (
            np.array(reference_trajectory), np.array(drone_trajectory),
            divergences, np.array(actions)
        )

    def compute_speed(self, drone_traj):
        """
        Compute speed, given a trajectory of drone positions
        """
        if len(drone_traj) == 0:
            return [0]
        dist = []
        for j in range(len(drone_traj) - 1):
            dist.append(
                (np.linalg.norm(drone_traj[j, :3] - drone_traj[j + 1, :3])) /
                self.dt
            )
        return [round(d, 2) for d in dist]

    def sample_circle(self):
        possible_planes = [[0, 1], [0, 2], [1, 2]]
        plane = possible_planes[np.random.randint(0, 3, 1)[0]]
        radius = np.random.rand() * 3 + 2
        direct = np.random.choice([-1, 1])
        circle_args = {"plane": plane, "radius": radius, "direction": direct}
        return circle_args

    def run_mpc_ref(
        self,
        reference: str,
        nr_test: int = 10,
        max_steps: int = 200,
        thresh_div=2,
        thresh_stable=2,
        **kwargs
    ):
        for _ in range(nr_test):
            _ = self.follow_trajectory(
                reference,
                max_nr_steps=max_steps,
                thresh_div=thresh_div,
                thresh_stable=thresh_stable,
                use_mpc_every=1
                # **circle_args
            )

    def run_eval(
        self,
        reference: str = "rand",
        nr_test: int = 10,
        max_steps: int = 251,
        thresh_div=1,
        thresh_stable=1,
        return_dict=False,
        **kwargs
    ):
        """
        Function to evaluate a trajectory multiple times
        """
        np.random.seed(42)
        if nr_test == 0:
            return 0, 0
        div, stable = [], []
        for i in range(nr_test):
            # print("Run %d/%d" % (i + 1, nr_test))
            # circle_args = self.sample_circle()
            reference_traj, drone_traj, divergences, actions = self.follow_trajectory(
                reference,
                max_nr_steps=max_steps,
                thresh_div=thresh_div,
                thresh_stable=thresh_stable
                # **circle_args
            )
            div.append(np.mean(divergences))
            # before take over
            stable.append(np.sum(np.array(divergences) < thresh_div))
            # print(np.mean(divergences), no_large_div)
            # no_large_div = np.where(np.array(divergences) > thresh_div)[0][0]
            # stable.append(len(drone_traj))

        # Output results
        # get maximum steps we could make (depends on speed)
        max_steps_stable = len(reference_traj)
        stable = np.array(stable)
        div = np.array(div)
        # ratio of runs that were stable
        ratio_stable = np.sum(stable == max_steps_stable) / len(stable)
        # get tracking error only for the runs that were completed
        div_of_full_runs = div[stable == max_steps_stable]
        avg_full = np.mean(div_of_full_runs) if len(div_of_full_runs) > 0 else np.nan
        std_full = np.std(div_of_full_runs) if len(div_of_full_runs) > 0 else np.nan
        overall_avg = np.mean(div) if div.size > 0 else np.nan
        overall_std = np.std(div) if div.size > 0 else np.nan
        print(
            "Average tracking error: %3.2f (%3.2f)" %
            (overall_avg, overall_std)
        )
        if 0 < ratio_stable < 1:
            print(
                "Average error of full runs: %3.2f (%3.2f)" %
                (avg_full, std_full)
            )
        print(
            "Ratio of stable runs: %3.2f" % (ratio_stable)
        )

        if return_dict:
            return {
                "avg_tracking_error": avg_full,
                "std_tracking_error": std_full,
                "ratio_stable": ratio_stable
            }
        return  np.mean(stable) if stable.size > 0 else np.nan, \
                np.std(stable) if stable.size > 0 else np.nan, \
                avg_full, \
                std_full, \
                overall_avg, \
                overall_std

    def collect_training_data(self, outpath="data/jan_2021.npy"):
        """
        Run evaluation but collect and save states as training data
        """
        data = []
        for _ in range(80):
            _, drone_traj = self.straight_traj(max_nr_steps=100)
            data.extend(drone_traj)
        for _ in range(20):
            # vary plane and radius
            possible_planes = [[0, 1], [0, 2], [1, 2]]
            plane = possible_planes[np.random.randint(0, 3, 1)[0]]
            radius = np.random.rand() + .5
            # run
            _, drone_traj = self.circle_traj(
                max_nr_steps=500, radius=radius, plane=plane
            )
            data.extend(drone_traj)
        data = np.array(data)
        print(data.shape)
        np.save(outpath, data)


def load_model(model_path, epoch=""):
    """
    Load model and corresponding parameters
    """
    # load std or other parameters from json
    net, param_dict = load_model_params(model_path, "model_quad", epoch=epoch)
    param_dict["self_play"] = 0
    dataset = QuadDataset(1, **param_dict)

    controller = NetworkWrapper(net.to(device), dataset, **param_dict)

    return controller, param_dict


if __name__ == "__main__":
    # make as args:
    parser = argparse.ArgumentParser("Model directory as argument")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="current_model",
        help="Directory of model"
    )
    parser.add_argument(
        "-e", "--epoch", type=str, default="", help="Saved epoch"
    )
    parser.add_argument(
        "-r", "--ref", type=str, default="rand", help="which trajectory"
    )
    parser.add_argument(
        "-a", "--eval", type=int, default=0, help="run evaluation for steps"
    )
    parser.add_argument(
        '-p',
        '--points',
        type=str,
        default=None,
        help="use predefined reference"
    )
    parser.add_argument(
        "-n", "--animate", action='store_true', help="animate 3D"
    )
    parser.add_argument(
        "-u", "--unity", action='store_true', help="unity rendering"
    )
    parser.add_argument(
        "-f", "--flightmare", action='store_true', help="Flightmare"
    )
    parser.add_argument(
        "-s", "--save_traj", action="store_true", help="save the trajectory"
    )
    args = parser.parse_args()

    DYNAMICS = "flightmare"

    # CONTROLLER - define and load controller
    model_path = os.path.join("trained_models", "quad", args.model)
    # MPC
    if model_path.split(os.sep)[-1] == "mpc":
        # mpc parameters:
        params = {"horizon": 10, "dt": .1}
        controller = MPC(dynamics=DYNAMICS, **params)
    # Neural controller
    else:
        controller, params = load_model(model_path, epoch=args.epoch)

    # PARAMETERS
    params["render"] = 0
    # params["dt"] = .05
    # params["max_drone_dist"] = 1
    params["speed_factor"] = .4
    modified_params = {}  # {"mass": 1}
    # {"rotational_drag": np.array([.1, .1, .1])}
    # {"mass": 1}
    # {"translational_drag": np.array([.7, .7, .7])}
    # {
    #     "mass": 1,
    #     "frame_inertia": np.array([2, 2, 3]),
    #     "kinv_ang_vel_tau": np.array([21, 21, 3.0])
    # }
    if len(modified_params) > 0:
        print("MODIFIED: ", modified_params)

    # DEFINE ENVIRONMENT
    if args.flightmare:
        environment = FlightmareWrapper(params["dt"], args.unity)
    else:
        # DYNAMICS
        dynamics = (
            FlightmareDynamics(modified_params=modified_params)
            if DYNAMICS == "flightmare" else SimpleDynamics()
        )
        environment = QuadRotorEnvBase(dynamics, params["dt"])

    # EVALUATOR
    evaluator = QuadEvaluator(controller, environment, test_time=1, **params)

    np.random.seed(42)
    # Specify arguments for the trajectory
    fixed_axis = 1
    traj_args = {
        "plane": [0, 2],
        "radius": 2,
        "direction": 1,
        "thresh_div": 5,
        "thresh_stable": 2,
        "duration": 10,
        "max_steps": int(1000 / (params["speed_factor"] * 10)) + 1
    }
    if args.points is not None:
        from neural_control.trajectory.predefined_trajectories import (
            collected_trajectories
        )
        traj_args["points_to_traverse"] = collected_trajectories[args.points]

    # RUN
    if args.unity:
        evaluator.eval_env.env.connectUnity()

    if args.eval > 0:
        # run_mpc_analysis(evaluator, system="quad")
        evaluator.run_eval(args.ref, nr_test=args.eval, **traj_args)
        exit()

    # run one trajectory
    reference_traj, drone_traj, divergences, _ = evaluator.follow_trajectory(
        args.ref, max_nr_steps=2000, use_mpc_every=1000, **traj_args
    )
    # Save trajectories
    if args.save_traj:
        os.makedirs("output_video", exist_ok=True)
        np.save(
            os.path.join("output_video", f"quad_ref_{args.model}.npy"),
            reference_traj
        )
        np.save(
            os.path.join("output_video", f"quad_traj_{args.model}.npy"),
            drone_traj
        )
    print("Average divergence", np.mean(divergences))
    if args.animate:
        animate_quad(
            reference_traj,
            [drone_traj],
            # uncomment to save video
            # savefile=os.path.join(model_path, 'video.mp4')
        )

    if args.unity:
        evaluator.eval_env.env.disconnectUnity()
