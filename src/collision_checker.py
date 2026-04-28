import numpy as np
from pathlib import Path
import subprocess
import os

from xbot2_interface import pyxbot2_interface as xbi
from xbot2_interface import pyxbot2_collision as xbc
from xbot2_interface.pyaffine3 import Affine3

class CollisionChecker:
    def __init__(self, urdf, srdf):
        self.model = xbi.ModelInterface2(urdf, srdf, 'pin')
        self.collision_model = xbc.CollisionModel(self.model)

    def add_pipe(self, name, cyl_radius, cyl_length, cyl_pos, cyl_ori):

        cyl = xbc.shape.Cylinder()
        cyl.radius = cyl_radius
        cyl.length = cyl_length
        self.collision_model.addCollisionShape(name, "world", cyl, Affine3(pos=cyl_pos, rot=cyl_ori), 
                                               ['end_effector_F', 
                                                'ee_F', 
                                                'L_6_F', 
                                                'J6_F_stator',
                                                'world'])


    def compute_collisions(self, q):


        self.model.q = q
        self.model.update()
        self.collision_model.update()

        coll_pairs = self.collision_model.getCollisionPairs(include_env=True)
        is_colliding, pair_ids = self.collision_model.checkCollision(include_env=True)

        return is_colliding, [coll_pairs[i] for i in pair_ids]