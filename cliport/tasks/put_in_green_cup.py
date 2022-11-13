"""Put Blocks in Bowl Task."""

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

import random
import pybullet as p


class PutInGreenCupUnSeenColors(Task):
    """Put Sugar/Coffee in Cup base class and task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put sugar and coffee in the green cup"
        self.task_completed_desc = "done placing sugar in cup."

    def reset(self, env):
        super().reset(env)
        # n_cups = np.random.randint(1, 4)
        # n_cubes = np.random.randint(2*n_cups, 4*n_cups)
        n_cups = 1
        n_cubes = 2

        all_color_names = self.get_colors()
        # cup_color_names = random.sample(all_color_names, 1)
        cup_color_names = ['green']
        colors = [utils.COLORS[cn] for cn in cup_color_names]
        sugar_color = utils.COLORS['white']
        coffee_color = utils.COLORS['brown']

        # Add bowls.
        cup_size = (0.08, 0.08, 0)
        cup_urdf = 'cup/cup.urdf'
        cup_poses = []
        for _ in range(n_cups):
            cup_pose = self.get_random_pose(env, cup_size)
            cup_id = env.add_object(cup_urdf, cup_pose, 'fixed')
            p.changeVisualShape(cup_id, -1, rgbaColor=colors[0] + [1])
            cup_poses.append(cup_pose)

        # Add sugar/coffee cupes.
        cubes = []
        sugar_size = (0.04, 0.04, 0.01)
        coffee_size = (0.03, 0.03, 0.01)
        cube_urdf = 'stacking/block.urdf'
        sugar_cube_urdf = 'stacking/sugar_cube.urdf'
        coffee_cube_urdf = 'stacking/coffee_cube.urdf'
        for i in range(n_cubes):
            # sugar first
            if i < n_cubes // 2:
                cube_color = sugar_color
                cube_size = sugar_size
                cube_urdf = sugar_cube_urdf
            else:
                cube_color = coffee_color
                cube_size = coffee_size
                cube_urdf = coffee_cube_urdf
            cube_pose = self.get_random_pose(env, cube_size)
            cube_id = env.add_object(cube_urdf, cube_pose)
            p.changeVisualShape(cube_id, -1, rgbaColor=cube_color + [1])
            cubes.append((cube_id, (0, None)))

        # Goal: put a sugar and a coffee cupe in the colored cup.
        # objs, matches, targs, replace, rotations, _, _, _ = goal
        self.goals.append(([cubes[0]], np.ones((1, 1)),
                           cup_poses, False, True, 'pose', None, 0.5))
        self.lang_goals.append(self.lang_template)
        self.goals.append(([cubes[1]], np.ones((1, 1)),
                           cup_poses, False, True, 'pose', None, 0.5))
        self.lang_goals.append(self.lang_template)

        # Only one mistake allowed.
        self.max_steps = len(cubes) + 1

        # Colors of distractor objects.
        distractor_cup_colors = [utils.COLORS[c] for c in utils.COLORS if c not in cup_color_names]
        distractor_cube_colors = [utils.COLORS[c] for c in utils.COLORS if c not in [sugar_color, coffee_color]]

        # Add distractors.
        n_distractors = 0
        max_distractors = 6
        while n_distractors < max_distractors:
            is_cupe = np.random.rand() > 0.5
            if is_cupe:
                colors = distractor_cube_colors
                is_sugar = np.random.rand() > 0.5
                size = sugar_size if is_sugar else coffee_size
                urdf = sugar_cube_urdf if is_sugar else coffee_cube_urdf
            else:
                urdf = cup_urdf
                colors = distractor_cup_colors
                size = cup_size

            pose = self.get_random_pose(env, size)
            if not pose:
                continue
            obj_id = env.add_object(urdf, pose)
            color = colors[n_distractors % len(colors)]
            if not obj_id:
                continue
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            n_distractors += 1

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PutInGreenCupSeenColors(PutInGreenCupUnSeenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS


class PutInGreenCupFull(PutInGreenCupUnSeenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
        return all_colors