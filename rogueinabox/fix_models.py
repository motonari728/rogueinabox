import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, initializers, Input, MaxPooling2D, Lambda
from keras.layers import Conv2D, ZeroPadding2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam, RMSprop
from abc import ABC, abstractmethod

import skimage
from skimage import transform, exposure

from models import ModelReshaper, ModelBuilder


class Fix_Ml_Nr_ModelReshaper(ModelReshaper):
    def __init__(self, rogue_box, layers):
        self.rows = 43
        self.columns = 159
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


class Fix_2T_ModelBuilder(ModelBuilder):
    def __init__(self, shape):
        super().__init__()
        self.depth = 2
        self.shape = shape
    
    def build_model(self):
        initializer = initializers.random_normal(stddev=0.02)
    
        input_img = Input(shape=self.shape)
        # 21,79周辺1マスを切り出す. 3*3
        # output_shapeはtheano用。tehsorflowなら自動推論される
        input_2 = Lambda(lambda x: x[:, :, 20:22, 78:80], output_shape=lambda x: (None, self.layers, 3, 3))(input_img) 

        # whole map tower
        tower_1 = Conv2D(64, (3, 3), data_format="channels_first", strides=(1, 1), kernel_initializer=initializer, padding="same")(input_img)
        tower_1 = Conv2D(32, (3, 3), data_format="channels_first", strides=(1, 1), kernel_initializer=initializer, padding="same")(tower_1)
        tower_1 = MaxPooling2D(pool_size=(22, 80), data_format="channels_first")(tower_1)





class Fix_T_5L_Ml_Nr_ModelManager(Fix_Ml_Nr_ModelReshaper, T_ModelBuilder):
    def __init__(self, rogue_box):
        layers = 5
        Fix_Ml_Nr_ModelReshaper.__init__(self, rogue_box, layers)
        T_ModelBuilder.__init__(self, self._shape)