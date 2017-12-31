import numpy as np
import scipy
import itertools
from abc import ABC, abstractmethod

from states import M_P_D_S_Sn_StateGenerator

class fix_M_P_D_S_Sn_StateGenerator(M_P_D_S_Sn_StateGenerator):
    def _set_shape(self):
        self._shape = (5, 22*2, 80*2)

    def compute_state(self):
        state = super().compute_state()
        state = convert_to_fix(state, self._shape, self.positions["player_pos"])
        return state


def convert_to_fix(state, shape, player_pos):
    new_state = np.zeros(shape, dtype=np.uint8)
    if player_pos == []:
        p_i, p_j = player_pos
        # p_i, p_jを中心に移したときの画像左上の位置
        x = 21 - p_i
        y = 79 - p_j
        new_state[:, x:x+22, y:y+80] = state
    return new_state

























