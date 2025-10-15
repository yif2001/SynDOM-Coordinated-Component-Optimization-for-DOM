import numpy as np
import pyflex
from copy import deepcopy
from softgym.envs.cloth_env import ClothEnv
from softgym.utils.pyflex_utils import center_object


class ClothFoldEnv(ClothEnv):
    def __init__(self, cached_states_path='cloth_fold_init_states.pkl', **kwargs):
        self.fold_group_a = self.fold_group_b = None
        self.init_pos, self.prev_dist = None, None
        self.num_variations = kwargs['num_variations']

        #=============Initialize corner_pick====3yue11ri xiugai==============
        # 初始化 corner_pick
        #self.corner_pick = kwargs.get('corner_pick', [0, 0, 0])  # 根据需求设置（例如 [x, y, z]）
        #self.corner_pick = kwargs.get('corner_pick', [0, 0])  # 设置默认值，可以根据具体需要调整
        # =============Initialize corner_pick====3yue11ri xiugai====/////////


        # =============Initialize corner_pick====3yue11 night add==============
        # Store the actual positions of the corners to pick
        self.corner_pick = []

        # =============Initialize corner_pick====3yue11 night add==////////////


        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

        # RSI and IR
        self.enable_rsi = kwargs.get('enable_rsi', False)
        self.enable_rsi_ir = False
        self.rsi_ir_prob = kwargs.get('rsi_ir_prob', 0)
        rsi_file = kwargs.get('rsi_file', None)
        if rsi_file is not None:
            self.reference_next_state_info = np.load(rsi_file, allow_pickle=True)
        self.reference_next_state_info_ep = None
        self.reference_next_action_info_ep = None
        self.chosen_step = 0
        self.non_rsi_ir = kwargs.get('non_rsi_ir', False)
        self.enable_action_matching = kwargs.get('enable_action_matching', False)
        self.enable_loading_states_from_folder = kwargs.get('enable_loading_states_from_folder', False)
        self.ep_task_reward = self.ep_il_reward = 0. # initialize

    def rotate_particles(self, angle):
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos -= center
        new_pos = pos.copy()
        new_pos[:, 0] = (np.cos(angle) * pos[:, 0] - np.sin(angle) * pos[:, 2])
        new_pos[:, 2] = (np.sin(angle) * pos[:, 0] + np.cos(angle) * pos[:, 2])
        new_pos += center
        pyflex.set_positions(new_pos)

    def _sample_cloth_size(self):
        if self.num_variations == 1:
            return 80, 80
        else:
            return np.random.randint(60, 120), np.random.randint(60, 120)

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        particle_radius = self.cloth_particle_radius
        if self.action_mode in ['sawyer', 'franka']:
            cam_pos, cam_angle = np.array([0.0, 1.62576, 1.04091]), np.array([0.0, -0.844739, 0])
        else:
            if self.num_variations == 1:
                # top down view only for sim-to-real experiments
                cam_pos, cam_angle = np.array([-0.0, 1.0, 0]), np.array([0, -90 / 180. * np.pi, 0.])
            else:
                 # default side view
                cam_pos, cam_angle = np.array([-0.0, 0.82, 0.82]), np.array([0, -45 / 180. * np.pi, 0.])
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [int(0.6 / particle_radius), int(0.368 / particle_radius)],
            'ClothStiff': [0.8, 1, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': cam_pos,
                                   'angle': cam_angle,
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'flip_mesh': 0,
        }

        return config

    def generate_env_variation(self, num_variations=2, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 1000  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.2  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()
        default_config['flip_mesh'] = 1

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']

            self.set_scene(config)
            self.action_tool.reset([0., -1., 0.])
            pos = pyflex.get_positions().reshape(-1, 4)
            pos[:, :3] -= np.mean(pos, axis=0)[:3]
            if self.action_mode in ['sawyer', 'franka']: # Take care of the table in robot case
                pos[:, 1] = 0.57
            else:
                pos[:, 1] = 0.005
            pos[:, 3] = 1
            pyflex.set_positions(pos.flatten())
            pyflex.set_velocities(np.zeros_like(pos))
            for _ in range(5):  # In case if the cloth starts in the air
                pyflex.step()

            for wait_i in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
                    break

            center_object()
            angle = (np.random.random() - 0.5) * np.pi / 2
            if self.num_variations != 1:
                self.rotate_particles(angle)

            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def set_test_color(self, num_particles):
        """
        Assign random colors to group a and the same colors for each corresponding particle in group b
        :return:
        """
        colors = np.zeros((num_particles))
        rand_size = 30
        rand_colors = np.random.randint(0, 5, size=rand_size)
        rand_index = np.random.choice(range(len(self.fold_group_a)), rand_size)
        colors[self.fold_group_a[rand_index]] = rand_colors
        colors[self.fold_group_b[rand_index]] = rand_colors
        self.set_colors(colors)

    def reset(self, config=None, initial_state=None, config_id=None, is_eval=False):
        if config is None:
            if config_id is None:
                if self.eval_flag:
                    eval_beg = int(0.8 * len(self.cached_configs))
                    config_id = np.random.randint(low=eval_beg, high=len(self.cached_configs)) if not self.deterministic else eval_beg
                else:
                    train_high = int(0.8 * len(self.cached_configs))
                    config_id = np.random.randint(low=0, high=max(train_high, 1)) if not self.deterministic else 0

            self.current_config = self.cached_configs[config_id]
            self.current_config_id = config_id
            self.set_scene(self.cached_configs[config_id], self.cached_init_states[config_id])
        else:
            self.current_config = config
            self.set_scene(config, initial_state)
        self.particle_num = pyflex.get_n_particles()
        self.prev_reward = 0.
        self.ep_il_reward = 0.
        self.ep_task_reward = 0.
        self.time_step = 0
        self.chosen_step = 0

        if (not is_eval) and self.enable_rsi and np.random.uniform(0,1) <= self.rsi_ir_prob:
            self.reset_to_state() # reset chosen_step and chosen state
            self.enable_rsi_ir = True
        else:
            self.enable_rsi_ir = False

        obs = self._reset()
        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))
        return obs

    def reset_to_state(self):
        '''
        RSI
        '''
        state_trajs = self.reference_next_state_info['state_trajs']
        configs = self.reference_next_state_info['configs']
        ob_trajs = self.reference_next_state_info['ob_trajs']
        action_trajs = self.reference_next_state_info['action_trajs']
        reference_state_index = np.random.randint(0, len(ob_trajs))
        self.reference_next_state_info_ep = ob_trajs[reference_state_index]
        self.reference_next_action_info_ep = action_trajs[reference_state_index]
        self.chosen_step = np.random.randint(0, len(self.reference_next_state_info_ep))
        self.time_step = self.chosen_step

        # reset environment
        self.current_config = configs[reference_state_index]
        if self.enable_loading_states_from_folder:
            self.set_scene(configs[reference_state_index], np.load(state_trajs[reference_state_index][self.chosen_step], allow_pickle=True).item())
        else:
            self.set_scene(configs[reference_state_index], state_trajs[reference_state_index][self.chosen_step])

    def _reset(self):
        """ Right now only use one initial state. Need to make sure _reset always give the same result. Otherwise CEM will fail."""
        # 判断当前环境类型
        env_type = getattr(self, 'env_name', self.__class__.__name__)
        is_diagonal = 'DiagonalPinned' in env_type or 'DiagonalUnpinned' in env_type
        is_pinned = 'DiagonalPinned' in env_type
        
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            p1, p2, p3, p4 = self._get_key_point_idx()
            key_point_pos = particle_pos[(p1, p2), :3]  # Was changed from from p1, p4.
            middle_point = np.mean(key_point_pos, axis=0)
            self.action_tool.reset([middle_point[0], 0.1, middle_point[2]])

        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1],
                                                                        config['ClothSize'][0])  # Reversed index here

        # 对于对角线折叠，分组方式不同
        if is_diagonal:
            cloth_dimx = config['ClothSize'][0]
            cloth_dimy = config['ClothSize'][1]
            
            # 对角线折叠的分组
            if cloth_dimx == cloth_dimy:  # 正方形布料
                # 沿对角线划分
                fold_group_a_indices = []
                fold_group_b_indices = []
                
                for i in range(cloth_dimy):
                    for j in range(cloth_dimx):
                        if i + j < cloth_dimx:  # 左下三角形
                            fold_group_a_indices.append(particle_grid_idx[i, j])
                        else:  # 右上三角形
                            fold_group_b_indices.append(particle_grid_idx[i, j])
                
                self.fold_group_a = np.array(fold_group_a_indices)
                self.fold_group_b = np.array(fold_group_b_indices)
            else:  # 长方形布料
                # 根据对角线近似划分
                diagonal_slope = cloth_dimy / cloth_dimx
                fold_group_a_indices = []
                fold_group_b_indices = []
                
                for i in range(cloth_dimy):
                    for j in range(cloth_dimx):
                        if i < diagonal_slope * j:  # 上方
                            fold_group_b_indices.append(particle_grid_idx[i, j])
                        else:  # 下方
                            fold_group_a_indices.append(particle_grid_idx[i, j])
                
                self.fold_group_a = np.array(fold_group_a_indices)
                self.fold_group_b = np.array(fold_group_b_indices)
                
            # 如果是固定点版本，标记固定点
            if is_pinned:
                # 标记四个角点，或者布料的顶部边缘
                self.pinned_points = [
                    particle_grid_idx[0, 0],       # 左上角
                    particle_grid_idx[0, -1]       # 右上角
                ]
        else:
            # 原始标准折叠的分组
            cloth_dimx = config['ClothSize'][0]
            x_split = cloth_dimx // 2
            self.fold_group_a = particle_grid_idx[:, :x_split].flatten()
            self.fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()

        colors = np.zeros(num_particles)
        colors[self.fold_group_a] = 1
        # self.set_colors(colors) # TODO the phase actually changes the cloth dynamics so we do not change them for now. Maybe delete this later.

        pyflex.step()
        self.init_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pos_a = self.init_pos[self.fold_group_a, :]
        pos_b = self.init_pos[self.fold_group_b, :]
        self.prev_dist = np.mean(np.linalg.norm(pos_a - pos_b, axis=1))

        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']

        particle_pos = pyflex.get_positions().reshape(-1, 4)  # 确保我们有更新的粒子位置

        self.x_min = np.amin(particle_pos[:, 0])
        self.x_max = np.amax(particle_pos[:, 0])

        self.z_min = np.amin(particle_pos[:, 2])
        self.z_max = np.amax(particle_pos[:, 2])

        self.y_coor = particle_pos[0, 1]  # checked and saw that these were all the same

        # 获取角点索引
        corner1_idx = particle_grid_idx[0, 0]       # 左上
        corner2_idx = particle_grid_idx[0, -1]      # 右上
        corner3_idx = particle_grid_idx[-1, 0]      # 左下
        corner4_idx = particle_grid_idx[-1, -1]     # 右下

        self.corner_pick_idx = []
        self.corner_drop_idx = []

        # 对于对角线折叠，拾取和放置角点的选择有所不同
        if is_diagonal:
            # 对角线折叠：左下角到右上角
            if corner3_idx in self.fold_group_a:
                self.corner_pick_idx.append(corner3_idx)  # 左下角
            if corner2_idx in self.fold_group_b:
                self.corner_drop_idx.append(corner2_idx)  # 右上角
                
            # 根据是否固定来确定其他角点
            if is_pinned:
                # 固定点版本：上边缘是固定的
                if corner1_idx not in self.corner_pick_idx and corner1_idx not in self.corner_drop_idx:
                    self.corner_drop_idx.append(corner1_idx)  # 左上角也是放置点
                
                if corner4_idx not in self.corner_pick_idx and corner4_idx not in self.corner_drop_idx:
                    self.corner_pick_idx.append(corner4_idx)  # 右下角也是拾取点
            else:
                # 非固定点版本：更灵活的折叠
                if corner1_idx in self.fold_group_a and corner1_idx not in self.corner_pick_idx:
                    self.corner_pick_idx.append(corner1_idx)
                elif corner1_idx not in self.corner_drop_idx:
                    self.corner_drop_idx.append(corner1_idx)
                    
                if corner4_idx in self.fold_group_b and corner4_idx not in self.corner_drop_idx:
                    self.corner_drop_idx.append(corner4_idx)
                elif corner4_idx not in self.corner_pick_idx:
                    self.corner_pick_idx.append(corner4_idx)
        else:
            # 标准折叠的角点分配
            # test to see which cloth folding group to place each corner in
            if corner1_idx in self.fold_group_a:
                self.corner_pick_idx.append(corner1_idx)
            else:
                self.corner_drop_idx.append(corner1_idx)

            if corner2_idx in self.fold_group_a:
                self.corner_pick_idx.append(corner2_idx)
            else:
                self.corner_drop_idx.append(corner2_idx)

            if corner3_idx in self.fold_group_a:
                self.corner_pick_idx.append(corner3_idx)
            else:
                self.corner_drop_idx.append(corner3_idx)

            if corner4_idx in self.fold_group_a:
                self.corner_pick_idx.append(corner4_idx)
            else:
                self.corner_drop_idx.append(corner4_idx)

        # 确保 corner_pick 和 corner_drop 填充了实际位置
        self.corner_pick = []
        for idx in self.corner_pick_idx:
            self.corner_pick.append(particle_pos[idx][:3])

        self.corner_drop = []
        for idx in self.corner_drop_idx:
            self.corner_drop.append(particle_pos[idx][:3])

        # 标准折叠的优化处理
        if not is_diagonal and len(self.corner_pick_idx) > 0 and len(self.corner_drop_idx) > 1:
            corner_pick0 = particle_pos[self.corner_pick_idx[0]][:3]
            corner_drop0 = particle_pos[self.corner_drop_idx[0]][:3]
            corner_drop1 = particle_pos[self.corner_drop_idx[1]][:3]

            # calculate distance between corner_pick and corner_drop, corner_drop
            dist0 = np.linalg.norm(corner_pick0 - corner_drop0)
            dist1 = np.linalg.norm(corner_pick0 - corner_drop1)

            # if this is the case swap so we don't fold diagonally
            if dist0 > dist1:
                # 交换索引
                tmp = self.corner_drop_idx[0]
                self.corner_drop_idx[0] = self.corner_drop_idx[1]
                self.corner_drop_idx[1] = tmp

                # 同时交换位置
                tmp = self.corner_drop[0]
                self.corner_drop[0] = self.corner_drop[1]
                self.corner_drop[1] = tmp

        self.expert_state = 0
        
        return self._get_obs()


    def step(self, action, record_continuous_video=False, img_size=None):
        """ If record_continuous_video is set to True, will record an image for each sub-step"""
        frames = []
        for i in range(self.action_repeat):
            self._step(action)
            if record_continuous_video and i % 2 == 0:  # No need to record each step
                frames.append(self.get_image(img_size, img_size))
        obs = self._get_obs()
        reward, rew_info = self.compute_reward(action, obs, set_prev_reward=True, rew_info=True)
        info = self._get_info()
        self.ep_il_reward += rew_info['il_reward']
        self.ep_task_reward += rew_info['task_reward']

        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))
        self.chosen_step += 1
        self.time_step += 1

        done = False
        if self.time_step >= self.horizon:
            done = True
        if record_continuous_video:
            info['flex_env_recorded_frames'] = frames
        return obs, reward, done, info

    def _step(self, action):
        self.action_tool.step(action)
        if self.action_mode in ['sawyer', 'franka']:
            print(self.action_tool.next_action)
            pyflex.step(self.action_tool.next_action)
        else:
            pyflex.step()

    def _get_obs_key_points(self):
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        keypoint_pos = particle_pos[self._get_key_point_idx(), :3]
        pos = keypoint_pos

        if self.action_mode in ['sphere', 'picker', 'pickerpickandplace']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])
        return pos

    def compute_reward(self, action=None, obs=None, set_prev_reward=False, expert_action=None, rew_info=False):
        """
        阶段性奖励函数，根据任务进度动态调整奖励权重，并添加目标导向奖励
        版本：2025/4/27 兼容对角折叠
        """
        pos = pyflex.get_positions().reshape(-1, 4)[:, :3]  # 获取当前所有粒子的位置
        
        # 判断当前环境类型
        env_type = getattr(self, 'env_name', self.__class__.__name__)
        is_diagonal = 'DiagonalPinned' in env_type or 'DiagonalUnpinned' in env_type
        is_pinned = 'DiagonalPinned' in env_type
        
        # 基础奖励计算
        pos_group_a = pos[self.fold_group_a]
        pos_group_b = pos[self.fold_group_b]
        pos_group_b_init = self.init_pos[self.fold_group_b]
        group_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1))
        fixation_dist = np.mean(np.linalg.norm(pos_group_b - pos_group_b_init, axis=1))
        
        # 计算任务进度 - 基于group_dist相对于初始距离的减少比例
        progress = 1.0 - min(1.0, group_dist / max(0.001, self.prev_dist))
        
        # 1. 角点对齐奖励：计算目标角点与当前角点的距离
        target_corners = [self.init_pos[idx] for idx in self.corner_drop_idx]
        current_corners = pos[self.corner_drop_idx]
        corner_dists = [np.linalg.norm(t - c) for t, c in zip(target_corners, current_corners)]
        corner_reward = -np.mean(corner_dists)
        
        # 2. 平整度奖励：计算布料表面的标准差
        heights = pos[:, 1]
        flatness_reward = -np.std(heights)
        
        # 3. 褶皱惩罚：检测局部区域的不平整度
        wrinkle_penalty = self._compute_wrinkle_penalty(pos)
        
        # 4. 对称性奖励：折叠后应保持对称
        symmetry_reward = self._compute_symmetry(pos)
        
        # 根据不同任务类型调整权重
        if is_diagonal:
            # 对角线折叠任务的权重调整
            # 对角线折叠需要更高的精度和控制
            corner_weight = 0.5 * (1.0 - progress)  # 增加角点权重
            flatness_weight = 0.4 * (0.5 - abs(0.5 - progress))
            wrinkle_weight = 0.4 * (0.5 - abs(0.5 - progress))
            symmetry_weight = 0.5 * progress
            
            # 对固定点任务，减小fixation_weight以减轻对固定点的惩罚
            if is_pinned:
                fixation_weight = 0.8 * (0.5 + 0.5 * progress)
                group_dist_weight = 1.2 + 0.6 * progress
            else:
                fixation_weight = 1.3 * (0.5 + 0.5 * progress)  # 非固定点需要更强的位置控制
                group_dist_weight = 1.0 + 0.5 * progress
        else:
            # 原始折叠任务的权重
            corner_weight = 0.4 * (1.0 - progress)
            flatness_weight = 0.3 * (0.5 - abs(0.5 - progress))
            wrinkle_weight = 0.3 * (0.5 - abs(0.5 - progress))
            symmetry_weight = 0.4 * progress
            fixation_weight = 1.2 * (0.5 + 0.5 * progress)
            group_dist_weight = 1.0 + 0.5 * progress
        
        # 基础折叠奖励
        fold_reward = -group_dist_weight * group_dist - fixation_weight * fixation_dist
        
        # 辅助奖励
        alignment_reward = (corner_weight * corner_reward + 
                            flatness_weight * flatness_reward + 
                            wrinkle_weight * wrinkle_penalty +
                            symmetry_weight * symmetry_reward)
        
        # 目标导向奖励
        goal_reward = self._compute_goal_reward(obs)
        
        # 完成奖励 - 当接近完成时给予额外奖励
        # 对角线折叠需要更严格的完成条件
        completion_bonus = 0.0
        if is_diagonal:
            if group_dist < 0.04 and fixation_dist < (0.04 if is_pinned else 0.02):
                completion_bonus = 2.5  # 增加完成奖励
        else:
            if group_dist < 0.05 and fixation_dist < 0.03:
                completion_bonus = 2.0
        
        # 组合所有奖励
        reward = fold_reward + alignment_reward + goal_reward + completion_bonus
        
        # 对角线折叠可能需要额外的形状奖励
        if is_diagonal:
            # 计算对角线折叠的特殊形状奖励
            diagonal_shape_reward = self._compute_diagonal_shape_reward(pos) if hasattr(self, '_compute_diagonal_shape_reward') else 0.0
            reward += 0.3 * diagonal_shape_reward
        
        # 模仿学习奖励（如启用）
        il_reward = 0
        if self.enable_rsi_ir and self.chosen_step < len(self.reference_next_state_info_ep):
            obs_key_points = self._get_obs_key_points()
            ref_dist = np.linalg.norm(self.reference_next_state_info_ep[self.chosen_step] - obs_key_points)
            il_reward = 1/(1 + np.exp(ref_dist))
            reward += il_reward
        
        if rew_info:
            reward_info = {
                'il_reward': il_reward if self.enable_rsi_ir else 0,
                'task_reward': fold_reward,
                'alignment_reward': alignment_reward,
                'goal_reward': goal_reward,
                'completion_bonus': completion_bonus,
                'progress': progress
            }
            
            # 为对角线任务添加额外信息
            if is_diagonal and hasattr(self, '_compute_diagonal_shape_reward'):
                reward_info['diagonal_shape_reward'] = diagonal_shape_reward
                
            return reward, reward_info
        return reward

    def _compute_wrinkle_penalty(self, pos):
        """计算布料上的褶皱惩罚"""
        config = self.get_current_config()
        cloth_size = config['ClothSize']
        
        # 重构布料网格以便分析
        try:
            grid_pos = pos.reshape(cloth_size[1], cloth_size[0], 3)
            wrinkle_score = 0.0
            
            # 简化版：使用局部高度变化检测褶皱
            for i in range(1, cloth_size[1]-1):
                for j in range(1, cloth_size[0]-1):
                    # 计算当前点与其邻居的高度差
                    center_height = grid_pos[i, j, 1]
                    neighbor_heights = [
                        grid_pos[i-1, j, 1], grid_pos[i+1, j, 1],
                        grid_pos[i, j-1, 1], grid_pos[i, j+1, 1]
                    ]
                    height_diffs = [abs(center_height - h) for h in neighbor_heights]
                    wrinkle_score += sum(height_diffs)
                    
            # 标准化
            wrinkle_score = -wrinkle_score / ((cloth_size[1]-2) * (cloth_size[0]-2) * 4)
            return wrinkle_score
        except:
            # 如果遇到问题（如形状不匹配），返回0
            return 0.0

    def _compute_symmetry(self, pos):
        """计算折叠后布料的对称性"""
        pos_group_a = pos[self.fold_group_a]
        pos_group_b = pos[self.fold_group_b]
        
        # 计算每对对应点的y坐标差异（折叠轴方向）
        y_diffs = np.abs(pos_group_a[:, 1] - pos_group_b[:, 1])
        
        # 对称时，y差异应该接近布料厚度
        ideal_y_diff = 0.01  # 估计布料厚度
        symmetry_score = -np.mean(np.abs(y_diffs - ideal_y_diff))
        
        return symmetry_score

    def _compute_goal_reward(self, obs):
        """基于目标状态计算奖励"""
        # 获取目标状态表示
        goal_state = self._get_goal_state()
        
        # 获取当前关键点状态
        current_keypoints = self._get_obs_key_points()
        
        # 计算当前状态与目标状态的距离
        keypoint_dist = np.linalg.norm(current_keypoints - goal_state['keypoints'])
        
        # 使用指数奖励函数，随着接近目标迅速增加奖励
        goal_reward = 1.0 * np.exp(-3.0 * keypoint_dist)
        
        return goal_reward

    def _get_goal_state(self):
        """获取理想的折叠状态表示 版本25/4/27 兼容对角折叠"""
        # 判断当前环境类型
        env_type = getattr(self, 'env_name', self.__class__.__name__)
        is_diagonal = 'DiagonalPinned' in env_type or 'DiagonalUnpinned' in env_type
        is_pinned = 'DiagonalPinned' in env_type
        
        # 如果已经计算过目标状态，直接返回
        goal_state_key = f'goal_state_{env_type}' if is_diagonal else 'goal_state'
        if hasattr(self, goal_state_key) and getattr(self, goal_state_key) is not None:
            return getattr(self, goal_state_key)
        
        # 保存当前状态
        orig_state = self.get_state()
        
        # 手动设置到理想折叠状态
        if is_diagonal:
            # 对角线折叠使用特殊的折叠方法
            perfect_fold_performance = self._set_to_folded_diagonal(is_pinned) if hasattr(self, '_set_to_folded_diagonal') else self._set_to_folded()
        else:
            # 标准折叠
            perfect_fold_performance = self._set_to_folded()
        
        # 获取关键点表示
        goal_keypoints = self._get_obs_key_points()
        
        # 恢复原始状态
        self.set_state(orig_state)
        
        # 存储目标状态
        goal_state = {
            'keypoints': goal_keypoints,
            'performance': perfect_fold_performance
        }
        
        # 根据任务类型存储不同的目标状态
        if is_diagonal:
            setattr(self, goal_state_key, goal_state)
        else:
            self.goal_state = goal_state
        
        return goal_state

    def _get_info(self):
        # Duplicate of the compute reward function!
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]
        pos_group_a = pos[self.fold_group_a]
        pos_group_b = pos[self.fold_group_b]
        pos_group_b_init = self.init_pos[self.fold_group_b]
        group_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1))
        fixation_dist = np.mean(np.linalg.norm(pos_group_b - pos_group_b_init, axis=1))
        performance = -group_dist - 1.2 * fixation_dist
        performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance
        info = {
            'performance': performance,
            'normalized_performance': (performance - performance_init) / (0. - performance_init),
            'neg_group_dist': -group_dist,
            'neg_fixation_dist': -fixation_dist
        }
        if 'qpg' in self.action_mode:
            info['total_steps'] = self.action_tool.total_steps
        return info

    def _set_to_folded(self):
        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]
        x_split = cloth_dimx // 2
        fold_group_a = particle_grid_idx[:, :x_split].flatten()
        fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()

        curr_pos = pyflex.get_positions().reshape((-1, 4))
        curr_pos[fold_group_a, :] = curr_pos[fold_group_b, :].copy()
        curr_pos[fold_group_a, 1] += 0.05  # group a particle position made tcurr_pos[self.fold_group_b, 1] + 0.05e at top of group b position.

        pyflex.set_positions(curr_pos)
        for i in range(10):
            pyflex.step()
        return self._get_info()['performance']
    
    def compute_expert_action(self):
        """
        为ClothFoldEnv实现高质量的专家策略
        该策略基于有限状态机，分阶段完成布料折叠任务
        状态：
        0 - 移动到第一个拾取角
        1 - 抓住角落
        2 - 抬起角落
        3 - 移动到放置位置
        4 - 放下角落
        5 - 释放角落
        6 - 移动到空中中立位置
        """
        # 获取当前picker和粒子的位置
        picker_pos, particle_pos = self.action_tool._get_pos()
        
        # 计算相关阈值和参数
        pick_threshold = 0.03  # 接近角落的阈值
        lift_height = 0.1     # 抬起高度
        move_speed = 0.8      # 移动速度因子
        
        # 根据当前状态决定动作
        if self.expert_state == 0:  # 移动到拾取角
            # 获取目标角的位置
            corner_pos = particle_pos[self.corner_pick_idx[0]][:3]
            
            # 计算方向向量和距离
            dir_to_corner = corner_pos - picker_pos[0, :3]
            dist_to_corner = np.linalg.norm(dir_to_corner)
            
            if dist_to_corner < pick_threshold:
                self.expert_state = 1  # 进入抓取状态
                action = np.array([0., 0., 0., 0.])  # 停留但不抓取
            else:
                # 归一化方向并设置移动速度
                if dist_to_corner > 0:
                    dir_to_corner = dir_to_corner / dist_to_corner * min(dist_to_corner, move_speed)
                action = np.array([dir_to_corner[0], dir_to_corner[1], dir_to_corner[2], 0.])
        
        elif self.expert_state == 1:  # 抓取角落
            self.expert_state = 2
            action = np.array([0., 0., 0., 1.])  # 抓取
        
        elif self.expert_state == 2:  # 抬起角落
            # 计算向上抬起的位置
            current_height = picker_pos[0, 1]
            target_height = self.y_coor + lift_height
            
            if current_height >= target_height - 0.01:
                self.expert_state = 3  # 进入移动状态
                action = np.array([0., 0., 0., 1.])  # 保持抓取
            else:
                # 向上移动
                lift_dir = np.array([0., min(target_height - current_height, move_speed), 0.])
                action = np.array([lift_dir[0], lift_dir[1], lift_dir[2], 1.])
        
        elif self.expert_state == 3:  # 移动到放置位置
            # 获取目标放置位置
            drop_pos = particle_pos[self.corner_drop_idx[0]][:3]
            drop_pos[1] = picker_pos[0, 1]  # 保持当前高度
            
            # 计算方向向量和距离
            dir_to_drop = drop_pos - picker_pos[0, :3]
            dist_to_drop = np.linalg.norm(dir_to_drop)
            
            if dist_to_drop < pick_threshold:
                self.expert_state = 4  # 进入放下状态
                action = np.array([0., 0., 0., 1.])  # 保持抓取
            else:
                # 归一化方向并设置移动速度
                if dist_to_drop > 0:
                    dir_to_drop = dir_to_drop / dist_to_drop * min(dist_to_drop, move_speed)
                action = np.array([dir_to_drop[0], dir_to_drop[1], dir_to_drop[2], 1.])
        
        elif self.expert_state == 4:  # 放下角落
            # 计算向下放置的位置
            current_height = picker_pos[0, 1]
            target_height = self.y_coor + 0.01  # 略高于布料表面
            
            if current_height <= target_height + 0.01:
                self.expert_state = 5  # 进入释放状态
                action = np.array([0., 0., 0., 1.])  # 保持抓取
            else:
                # 向下移动
                lower_dir = np.array([0., -min(current_height - target_height, move_speed), 0.])
                action = np.array([lower_dir[0], lower_dir[1], lower_dir[2], 1.])
        
        elif self.expert_state == 5:  # 释放角落
            self.expert_state = 6
            action = np.array([0., 0., 0., 0.])  # 释放
        
        elif self.expert_state == 6:  # 移动到空中中立位置
            # 计算中心位置
            cloth_center_x = (self.x_min + self.x_max) / 2
            cloth_center_z = (self.z_min + self.z_max) / 2
            neutral_pos = np.array([cloth_center_x, self.y_coor + lift_height, cloth_center_z])
            
            # 计算方向向量和距离
            dir_to_neutral = neutral_pos - picker_pos[0, :3]
            dist_to_neutral = np.linalg.norm(dir_to_neutral)
            
            if dist_to_neutral < 0.05:
                # 任务完成，保持当前状态
                action = np.array([0., 0., 0., 0.])
            else:
                # 归一化方向并设置移动速度
                if dist_to_neutral > 0:
                    dir_to_neutral = dir_to_neutral / dist_to_neutral * min(dist_to_neutral, move_speed)
                action = np.array([dir_to_neutral[0], dir_to_neutral[1], dir_to_neutral[2], 0.])
        
        else:  # 默认状态，不应该到达这里
            action = np.array([0., 0., 0., 0.])
        
        # 如果有多个picker，扩展动作数组
        if self.action_tool.num_picker > 1:
            additional_actions = np.zeros(4 * (self.action_tool.num_picker - 1))
            action = np.concatenate([action, additional_actions])
        
        return action

    
    def _set_to_folded_diagonal(self, is_pinned=False):
        """专门为对角线折叠任务设置理想折叠状态"""
        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        
        # 获取当前位置
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        
        # 对角线折叠：fold_group_a (左下区域) 应该移动到 fold_group_b (右上区域) 的位置
        for i, idx_a in enumerate(self.fold_group_a):
            # 找到对应的 fold_group_b 索引
            # 在对角线折叠中，我们需要找到最近的对应点
            if i < len(self.fold_group_b):
                idx_b = self.fold_group_b[i]
                curr_pos[idx_a, :3] = curr_pos[idx_b, :3].copy()
                curr_pos[idx_a, 1] += 0.05  # 略微抬高以避免穿透
        
        # 固定点处理
        if is_pinned and hasattr(self, 'pinned_points'):
            # 恢复固定点的原始位置
            for idx in self.pinned_points:
                curr_pos[idx, :3] = self.init_pos[idx, :3].copy()
        
        # 设置新位置
        pyflex.set_positions(curr_pos)
        
        # 允许物理引擎稳定化
        for i in range(10):
            pyflex.step()
        
        return self._get_info()['performance']

    def _compute_diagonal_shape_reward(self, pos):
        """计算对角线折叠的形状奖励"""
        # 对角线折叠的理想形状：折叠后沿对角线形成三角形
        
        # 计算布料当前的轮廓
        x_coords = pos[:, 0]
        z_coords = pos[:, 2]
        y_coords = pos[:, 1]
        
        # 计算布料的边界框
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        
        # 理想情况下，对角线折叠后的布料应该是三角形
        # 计算实际与理想形状的偏差
        
        # 1. 计算对角线上的点
        diagonal_points = []
        for i, p in enumerate(pos):
            # 找到接近对角线的点
            if abs((p[0] - x_min) / (x_max - x_min) - (p[2] - z_min) / (z_max - z_min)) < 0.1:
                diagonal_points.append(i)
        
        if not diagonal_points:
            return 0.0
        
        # 2. 计算对角线的高度一致性
        diagonal_heights = pos[diagonal_points, 1]
        height_deviation = -np.std(diagonal_heights)
        
        # 3. 计算三角形区域内点的高度一致性
        fold_group_a_heights = pos[self.fold_group_a, 1]
        fold_group_b_heights = pos[self.fold_group_b, 1]
        
        group_a_height_consistency = -np.std(fold_group_a_heights)
        group_b_height_consistency = -np.std(fold_group_b_heights)
        
        # 组合奖励
        shape_reward = (height_deviation + 0.5 * group_a_height_consistency + 0.5 * group_b_height_consistency) / 3.0
        
        return shape_reward
