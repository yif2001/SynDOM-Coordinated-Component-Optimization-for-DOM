import numpy as np
import pickle
import os.path as osp
import pyflex
from softgym.envs.rope_env import RopeNewEnv
from copy import deepcopy
from softgym.utils.pyflex_utils import random_pick_and_place, center_object
import math

class RopeFlattenEnv(RopeNewEnv):
    def __init__(self, cached_states_path='rope_flatten_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """

        super().__init__(**kwargs)
        self.prev_distance_diff = None
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

        # stack frames
        self.prev_obs = None
        self.prev_prev_obs = None

        # normalize observations
        self.enable_normalize_obs = kwargs.get('enable_normalize_obs', False)
        self.normalize_obs_file = kwargs.get('normalize_obs_file', None)
        if self.normalize_obs_file:
            normalize_obs_file = np.load(self.normalize_obs_file, allow_pickle=True)
            self.max_obs = normalize_obs_file.item()['max_obs']
            self.min_obs = normalize_obs_file.item()['min_obs']
        else:
            self.max_obs = None
            self.min_obs = None

        self.ep_task_reward = self.ep_il_reward = 0. # initialize
        
        # 添加新变量用于动作平滑度计算
        self.prev_action = None
        self.goal_state = None  # 用于存储目标状态
        

    def generate_env_variation(self, num_variations=1, config=None, save_to_file=False, **kwargs):
        """ Generate initial states. Note: This will also change the current states! """
        generated_configs, generated_states = [], []
        if config is None:
            config = self.get_default_config()
        default_config = config
        for i in range(num_variations):
            config = deepcopy(default_config)
            config['segment'] = self.get_random_rope_seg_num()
            self.set_scene(config)

            self.update_camera('default_camera', default_config['camera_params']['default_camera'])
            config['camera_params'] = deepcopy(self.camera_params)
            self.action_tool.reset([0., -1., 0.])

            random_pick_and_place(pick_num=4, pick_scale=0.005)
            center_object()
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def get_random_rope_seg_num(self):
        return np.random.randint(40, 41)

    def _reset(self):
        config = self.current_config
        self.rope_length = config['segment'] * config['radius'] * 0.5

        # set reward range
        self.reward_max = 0
        rope_particle_num = config['segment'] + 1
        self.key_point_indices = self._get_key_point_idx(rope_particle_num)

        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions().reshape([-1, 4])
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.1, cy])

        # set reward range
        self.reward_max = 0
        self.reward_min = -self.rope_length
        self.reward_range = self.reward_max - self.reward_min

        return self._get_obs()

    def _step(self, action):
        if self.action_mode.startswith('picker'):
            self.action_tool.step(action)
            pyflex.step()
        else:
            raise NotImplementedError
        return

    def _get_endpoint_distance(self):
        pos = pyflex.get_positions().reshape(-1, 4)
        p1, p2 = pos[0, :3], pos[-1, :3]
        return np.linalg.norm(p1 - p2).squeeze()

    def _get_obs_key_points(self):
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        keypoint_pos = particle_pos[self.key_point_indices, :3]
        pos = keypoint_pos.flatten()
        if self.action_mode in ['sphere', 'picker']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])
        return pos

    def step(self, action, record_continuous_video=False, img_size=None):
        """ If record_continuous_video is set to True, will record an image for each sub-step"""
        if self.non_rsi_ir:
            # Get expert action BEFORE stepping (it needs current particle positions)
            expert_action = self.compute_expert_action()
            lb, ub = self.action_space.low, self.action_space.high
            scaled_expert = lb + (expert_action + 1.) * 0.5 * (ub - lb)
            scaled_expert = np.clip(scaled_expert, lb, ub)

        frames = []
        for i in range(self.action_repeat):
            self._step(action)
            if record_continuous_video and i % 2 == 0:  # No need to record each step
                frames.append(self.get_image(img_size, img_size))
        obs = self._get_obs()

        if self.enable_normalize_obs:
            if self.max_obs is None and self.min_obs is None:
                self.max_obs = obs.copy()
                self.min_obs = obs.copy()
            else:
                self.max_obs = np.maximum(self.max_obs, obs)
                self.min_obs = np.minimum(self.min_obs, obs)
                # equation of normalization between -1 and 1: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
                obs = 2 * ((obs - self.min_obs) / (self.max_obs - self.min_obs)) - 1
                obs = np.nan_to_num(obs)

        if self.enable_stack_frames:
            single_obs = obs.copy()
            if self.prev_prev_obs is None:
                obs = np.array([obs, obs, self.prev_obs.copy()])
            else:
                obs = np.array([obs, self.prev_obs.copy(), self.prev_prev_obs.copy()])
            self.prev_prev_obs = self.prev_obs.copy()
            self.prev_obs = single_obs

        if self.non_rsi_ir:
            reward, rew_info = self.compute_reward(action, obs, set_prev_reward=True, expert_action=scaled_expert, rew_info=True)
        else:
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

        if self.enable_normalize_obs and self.max_obs is not None:
            obs = 2 * ((obs - self.min_obs) / (self.max_obs - self.min_obs)) - 1
            obs = np.nan_to_num(obs)

        if self.enable_stack_frames:
            self.prev_obs = obs.copy()
            self.prev_prev_obs = None
            obs = np.array([obs, obs, obs])

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
        self.set_scene(configs[reference_state_index], state_trajs[reference_state_index][self.chosen_step])

    def compute_reward(self, action=None, obs=None, set_prev_reward=False, expert_action=None, rew_info=False):
        """
        阶段性奖励函数，根据任务进度动态调整奖励权重
        
        :param action: action to be executed
        :param obs: current observation
        :param set_prev_reward: unused
        :param expert_action: expert action (if available). MUST be normalized and clipped to action_space.[lb, ub]
        :param rew_info: whether to return extra params in reward computation
        """
        
        '''print(f"Action type: {type(action)}, Obs type: {type(obs)}")
        if isinstance(action, dict):
            print(f"Action keys: {action.keys()}")
        if isinstance(obs, dict):
            print(f"Obs keys: {obs.keys()}")'''
        # 基础奖励计算 - 端点距离与绳长的差异
        curr_endpoint_dist = self._get_endpoint_distance()
        endpoint_diff = -np.abs(curr_endpoint_dist - self.rope_length)
        
        # 计算当前任务进度 (0-1范围)
        # 当endpoint_diff接近0时，进度接近1
        progress = np.clip((endpoint_diff + self.rope_length) / self.rope_length, 0, 1)
        
        # 新增奖励组件
        # 1. 绳子平滑度奖励 - 惩罚绳子过度弯曲
        smoothness_reward = self._compute_smoothness_reward()
        
        # 2. 绳子直线度奖励 - 奖励绳子形成直线
        straightness_reward = self._compute_straightness_reward()
        
        # 3. 避免剧烈动作的奖励 - 鼓励平稳操作
        action_smoothness = 0
        if action is not None and hasattr(self, 'prev_action') and self.prev_action is not None:
            # 确保action和prev_action是相同类型
            if isinstance(action, np.ndarray) and isinstance(self.prev_action, np.ndarray):
                action_diff = np.linalg.norm(action - self.prev_action)
                action_smoothness = -0.1 * action_diff
            elif isinstance(action, dict) and isinstance(self.prev_action, dict):
                # 处理字典类型的动作
                action_diff = 0
                for key in action:
                    if key in self.prev_action:
                        action_diff += np.linalg.norm(action[key] - self.prev_action[key])
                action_smoothness = -0.1 * action_diff
                
        # 保存当前动作供下次使用
        if action is not None:
            if isinstance(action, np.ndarray):
                self.prev_action = action.copy()
            elif isinstance(action, dict):
                self.prev_action = {k: v.copy() if hasattr(v, 'copy') else v for k, v in action.items()}
            else:
                self.prev_action = action
        
        # 4. 目标导向奖励 - 基于与理想状态的接近程度
        goal_reward = self._compute_goal_reward(obs)
        
        # 阶段性权重调整
        # 初期：注重端点抓取和拉直初期动作
        # 中期：注重绳子平滑度
        # 后期：注重保持直线状态和最终精确调整
        endpoint_weight = 1.0  # 基础权重恒定
        smoothness_weight = 0.3 * (1 - progress)  # 早期更重要
        straightness_weight = 0.5 * progress  # 后期更重要
        goal_weight = 0.5 * progress  # 后期更重要
        
        # 组合基础奖励
        task_reward = (endpoint_weight * endpoint_diff + 
                    smoothness_weight * smoothness_reward + 
                    straightness_weight * straightness_reward + 
                    action_smoothness + 
                    goal_weight * goal_reward)
        
        # 计算完成奖励 - 当接近目标状态时给予额外奖励
        completion_bonus = 0
        if curr_endpoint_dist > 0.95 * self.rope_length and curr_endpoint_dist < 1.05 * self.rope_length:
            completion_bonus = 1.0  # 一次性奖励
            
        task_reward += completion_bonus
        
        # 设置r为任务奖励
        r = task_reward
        
        # 添加模仿学习奖励(保留原有代码)
        il_reward = 0
        if self.enable_rsi_ir and self.chosen_step < len(self.reference_next_state_info_ep):
            if self.enable_stack_frames:
                if self.chosen_step == 0:
                    ref_next_state_obs = np.array([self.reference_next_state_info_ep[self.chosen_step+1].numpy(), \
                        self.reference_next_state_info_ep[self.chosen_step+1].numpy(), \
                        self.reference_next_state_info_ep[self.chosen_step].numpy()])
                elif self.chosen_step == len(self.reference_next_state_info_ep)-1:
                    ref_next_state_obs = np.array([self.reference_next_state_info_ep[self.chosen_step].numpy(), \
                        self.reference_next_state_info_ep[self.chosen_step-1].numpy()])
                    obs = obs.copy()[1:, :] # remove the goal state obs because reference states do have have this.
                else:
                    ref_next_state_obs = np.array([self.reference_next_state_info_ep[self.chosen_step+1].numpy(), \
                        self.reference_next_state_info_ep[self.chosen_step].numpy(), \
                        self.reference_next_state_info_ep[self.chosen_step-1].numpy()])
                ref_dist = np.linalg.norm(ref_next_state_obs - obs)
            else:
                obs_key_points = self._get_obs_key_points()
                ref_dist = np.linalg.norm(self.reference_next_state_info_ep[self.chosen_step] - obs_key_points)

            if self.enable_action_matching:
                # state_reward + action_reward
                state_reward = 1/(1 + math.exp(ref_dist))
                ref_act_dist = np.linalg.norm(self.reference_next_action_info_ep[self.chosen_step] - action)
                action_reward = 1/(1 + math.exp(ref_act_dist))
                il_reward = state_reward + action_reward
            else:
                # il_reward = np.exp(-1 * ref_dist) # range [0,1], disable due to nan issue when multiplication is used in np.exp
                # sigmoid with cutoff at x=0 because ref_dist cannot be negative, so range is [0, 0.5]
                il_reward = 1/(1 + np.exp(ref_dist))
        else:
            il_reward = 0
            if self.non_rsi_ir: # Compute reward for non-RSI-IR
                # Compute imitation reward
                expert_action = expert_action.reshape(2,4)
                action = action.reshape(2,4)
                exp_pick = np.array(expert_action[:, 3] > 0.5, dtype=np.float32)
                agent_pick = np.array(action[:, 3] > 0.5, dtype=np.float32)
                ispick_weight = 0.5 # TODO: random, need to tune
                action_distance = np.linalg.norm(expert_action[:,:3] - action[:,:3]) + ispick_weight * np.linalg.norm(exp_pick - agent_pick, ord=1)
                il_reward = -action_distance # TODO: Consider sigmoid, tanh, etc.
                
        # 添加模仿学习奖励到r
        r += il_reward
        
        if rew_info:
            return r, {
                'il_reward': il_reward, 
                'task_reward': task_reward,
                'endpoint_reward': endpoint_diff,
                'smoothness_reward': smoothness_reward,
                'straightness_reward': straightness_reward,
                'goal_reward': goal_reward,
                'completion_bonus': completion_bonus,
                'progress': progress
            }
        else:
            return r

    def compute_expert_action(self):
        """ Simple (suboptimal) expert: Always moves the endpoints away from each other
        No limit on speed or applied force
        """
        picker_pos, particle_pos = self.action_tool._get_pos()
        end1, end2 = particle_pos[0, :3], particle_pos[-1, :3]
        pick1, pick2 = picker_pos[0, :3], picker_pos[-1, :3] # picker positions
        pick_dist = np.linalg.norm(pick1 - pick2)
        do_pick_thresh = self.action_tool.picker_radius + self.action_tool.particle_radius + self.action_tool.picker_threshold

        dir1 = pick1 - pick2
        dir1 = dir1/(np.linalg.norm(dir1) + 1e-8) # normalize
        dir2 = -dir1

        ### SIMPLE PICKER
        # if close to end (within threshold): action = move-along-line with picker>0.5
        # else: pick1 go to end1, pick2 go to end2
        # PICKER1
        p_to_e1 = end1 - pick1
        if np.linalg.norm(p_to_e1) < do_pick_thresh:
            act1 = np.hstack([dir1, [1.]])
            if pick_dist > self.rope_length:
                act1 = np.hstack([dir1 * 0, [1.]]) # do nothing
        else:
            temp = p_to_e1/(np.linalg.norm(p_to_e1) + 1e-8)
            act1 = np.hstack([temp, 0.])

        # PICKER2
        p_to_e2 = end2 - pick2
        if np.linalg.norm(p_to_e2) < do_pick_thresh:
            act2 = np.hstack([dir2, [1.]])
            if pick_dist > self.rope_length:
                act2 = np.hstack([dir2 * 0, [1.]]) # do nothing
        else:
            temp = p_to_e2/(np.linalg.norm(p_to_e2) + 1e-8)
            act2 = np.hstack([temp, 0.])

        # Combine
        expert_action = np.hstack([act1, act2])
        return expert_action

    def _get_info(self):
        """获取环境状态信息，包括扩展指标"""
        curr_endpoint_dist = self._get_endpoint_distance()
        curr_distance_diff = -np.abs(curr_endpoint_dist - self.rope_length)
        
        # 基础性能指标
        performance = curr_distance_diff
        normalized_performance = (performance - self.reward_min) / self.reward_range
        
        # 计算进度
        progress = np.clip((performance + self.rope_length) / self.rope_length, 0, 1)
        
        # 添加额外指标
        smoothness = self._compute_smoothness_reward()
        straightness = self._compute_straightness_reward()
        
        # 计算绳子当前的长度总和（作为参考）
        pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
        current_length = 0
        for i in range(len(pos) - 1):
            current_length += np.linalg.norm(pos[i+1] - pos[i])
        
        # 判断是否成功完成任务
        success = (curr_endpoint_dist > 0.95 * self.rope_length and 
                curr_endpoint_dist < 1.05 * self.rope_length and
                straightness > -0.1)  # 相当直
        
        return {
            'performance': performance,
            'normalized_performance': normalized_performance,
            'end_point_distance': curr_endpoint_dist,
            'progress': progress,
            'smoothness': smoothness,
            'straightness': straightness,
            'current_rope_length': current_length,
            'success': float(success)
        }

    def _compute_smoothness_reward(self):
        """计算绳子平滑度奖励 - 惩罚过度弯曲"""
        pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
        
        # 跳过端点，分析内部粒子的曲率
        angles = []
        for i in range(1, len(pos)-1):
            v1 = pos[i] - pos[i-1]
            v2 = pos[i+1] - pos[i]
            
            # 计算向量之间的角度
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm * v2_norm < 1e-6:  # 避免除零
                continue
                
            cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
            # 裁剪值避免数值误差
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        # 根据角度计算平滑度 - 角度越接近π（180度），越平滑
        if not angles:
            return 0.0
        
        mean_angle = np.mean(angles)
        # 返回范围为[-1,0]的奖励，越平滑奖励越接近0
        smoothness = -(np.pi - mean_angle) / np.pi
        
        try:
            pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
            # 剩余代码...
            return smoothness
        except Exception as e:
            print(f"Error in _compute_smoothness_reward: {e}")
            return 0.0  # 返回默认值
        

    def _compute_straightness_reward(self):
        """计算绳子直线度奖励 - 奖励绳子形成直线"""
        pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
        
        # 计算端点之间的直线
        start_point = pos[0]
        end_point = pos[-1]
        line_vector = end_point - start_point
        line_length = np.linalg.norm(line_vector)
        
        if line_length < 1e-6:  # 避免除零
            return -1.0
        
        # 单位方向向量
        line_direction = line_vector / line_length
        
        # 计算每个点到直线的距离
        total_deviation = 0
        for i in range(1, len(pos)-1):  # 跳过端点
            point_vector = pos[i] - start_point
            # 点到直线的投影长度
            projection_length = np.dot(point_vector, line_direction)
            # 投影点在直线上的位置
            projection_point = start_point + projection_length * line_direction
            # 点到直线的距离
            distance = np.linalg.norm(pos[i] - projection_point)
            total_deviation += distance
        
        # 归一化偏离度并转换为奖励
        avg_deviation = total_deviation / (len(pos) - 2)
        normalized_deviation = avg_deviation / (self.rope_length / 2)  # 假设最大偏离是绳长的一半
        straightness = -np.clip(normalized_deviation, 0, 1)  # 范围为[-1,0]，越直奖励越接近0
        
        return straightness

    def _compute_goal_reward(self, obs):
        """计算基于目标状态的奖励"""
        # 获取目标状态
        goal_state = self._get_goal_state()
        
        # 检查观察值类型
        if isinstance(obs, dict):
            if 'key_point' in obs:
                current_obs = obs['key_point']
                goal_obs = goal_state['keypoints']
            elif 'image' in obs:
                # 如果是图像，可以计算图像特征的差异
                # 或者跳过图像奖励
                return 0.0
            else:
                # 未知字典格式，返回0奖励
                return 0.0
        else:
            # 假设是numpy数组
            current_obs = obs
            goal_obs = goal_state['obs']
        
        # 计算与目标状态的距离
        distance = np.linalg.norm(current_obs - goal_obs)
        
        # 使用指数衰减将距离转换为奖励
        goal_reward = np.exp(-2.0 * distance / len(current_obs))
        
        return goal_reward

    def _get_goal_state(self):
        """获取理想的拉直状态"""
        try:
            # 如果已经计算过目标状态，直接返回
            if hasattr(self, 'goal_state') and self.goal_state is not None:
                return self.goal_state
            
            # 保存当前状态
            orig_state = self.get_state()
            
            # 手动设置到理想拉直状态
            pos = pyflex.get_positions().reshape(-1, 4)
            vel = pyflex.get_velocities().reshape(-1, 3)
            
            # 获取端点位置
            start_point = pos[0, :3]
            end_point = pos[-1, :3]
            
            # 计算方向向量
            direction = end_point - start_point
            length = np.linalg.norm(direction)
            
            if length < 1e-6:  # 如果端点距离几乎为零，选择一个默认方向
                direction = np.array([1.0, 0.0, 0.0])
                length = 1.0
            else:
                direction = direction / length
            
            # 计算理想绳长
            ideal_segment_length = self.rope_length / (len(pos) - 1)
            
            # 重新设置每个粒子的位置为直线
            for i in range(len(pos)):
                ratio = i / (len(pos) - 1)  # 0到1之间的比例
                pos[i, :3] = start_point + ratio * self.rope_length * direction
                vel[i] = np.zeros(3)
            
            # 更新位置和速度
            pyflex.set_positions(pos.flatten())
            pyflex.set_velocities(vel.flatten())
            
            # 运行几步物理模拟以稳定状态
            for _ in range(5):
                pyflex.step()
            
            # 获取理想状态的观察
            ideal_obs = self._get_obs()
            ideal_keypoints = self._get_obs_key_points()
            
            # 恢复原始状态
            self.set_state(orig_state)
            
            # 存储目标状态
            self.goal_state = {
                'obs': ideal_obs,
                'keypoints': ideal_keypoints,
            }
            
            return self.goal_state
        except Exception as e:
            print(f"Error in _get_goal_state: {e}")
            # 返回空的目标状态
            return {'obs': np.zeros(1), 'keypoints': np.zeros(1)}