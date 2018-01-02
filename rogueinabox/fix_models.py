from models import ModelReshaper
import numpy as np


def convert_to_fix(state, shape, player_pos):
    ''' layer数の変更に対応できるようにshapeを渡す。shape=(layer, width, height) '''
    new_state = np.zeros(shape, dtype=np.uint8)
    if player_pos == []:
        p_i, p_j = player_pos
        # p_i, p_jを中心に移したときの画像左上の位置
        x = 21 - p_i
        y = 79 - p_j
        new_state[:, x:x+22, y:y+80] = state
    return new_state


class Fix_Ml_Nr_ModelReshaper(ModelReshaper):
    def __init__(self, rogue_box, layers):
        self.rows = 44
        self.columns = 160
        self.padding = True
        self.actions_num = len(rogue_box.get_actions())
        super().__init__(rogue_box, layers)


    # state_generatorから送られてくるshapeと一致させる
    # ModelReshaperのinitで一致しているかチェックされる
    def _set_shape(self):
        self._shape = (self.layers, self.rows, self.columns)

    def reshape_initial_state(self, first_frame):
        initial_agent_state = first_frame.reshape(1, first_frame.shape[0], first_frame.shape[1], first_frame.shape[2])
        return initial_agent_state

    # ndarray.reshape(1次元目の次元数、2次元目の...)
    # 配列[0]にそのまま入れて返す
    # 高速化の余地ありそう
    # layer, row, collum?
    def reshape_new_state(self, old_state, new_frame):
        new_agent_state = new_frame.reshape(1, new_frame.shape[0], new_frame.shape[1], new_frame.shape[2])
        return new_agent_state


class Fix_T_5L_Ml_Nr_ModelManager(Fix_Ml_Nr_ModelReshaper, T_ModelBuilder):
    def __init__(self, rogue_box):
        layers = 5
        Fix_Ml_Nr_ModelReshaper.__init__(self, rogue_box, layers)
        T_ModelBuilder.__init__(self)