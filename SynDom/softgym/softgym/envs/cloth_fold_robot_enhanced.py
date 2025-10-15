import numpy as np
import pyflex
from softgym.envs.cloth_fold_robot import ClothFoldRobotEnv

class ClothFoldRobotEnhancedEnv(ClothFoldRobotEnv):
    """增强版的ClothFoldRobotEnv，添加了优化的奖励函数和正则化技术"""
    
    def __init__(self, cached_states_path='cloth_fold_robot_init_states.pkl', **kwargs):
        # 调用父类的初始化
        super().__init__(cached_states_path=cached_states_path, **kwargs)
        
        # 初始化阶段性奖励的权重
        self.progress = 0.0
        self.alignment_weight = 0.3
        self.flatness_weight = 0.2
        self.edge_alignment_weight = 1.0
        self.corner_weight = 0.5
        
    def _reset(self):
        """扩展_reset方法以初始化阶段性奖励所需的属性"""
        obs = super()._reset()
        
        # 记录初始状态的度量值
        self.initial_edge_dist = self._compute_edge_distance()
        self.initial_corner_dist = self._compute_corner_distance()
        self.initial_height_std = self._compute_height_std()
        
        # 初始化奖励记忆
        self.last_task_reward = 0
        self.progress = 0.0
        
        return obs
    
    def _compute_edge_distance(self):
        """计算边缘之间的距离"""
        pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
        pos_left_edge = pos[self.left_edge_indices]
        pos_right_edge = pos[self.right_edge_indices]
        pos_top_edge = pos[self.top_edge_indices]
        pos_bottom_edge = pos[self.bottom_edge_indices]
        
        dist_edges = np.mean(np.linalg.norm(pos_left_edge - pos_right_edge, axis=1)) + \
                    np.mean(np.linalg.norm(pos_top_edge - pos_bottom_edge, axis=1))
        return dist_edges
    
    def _compute_corner_distance(self):
        """计算角点与目标位置的距离"""
        pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
        pos_bottom_left_corner_cloth = pos[self.bottom_left_corner_pos]
        pos_top_right_corner_box = pos[self.top_right_corner_pos].copy()
        pos_top_right_corner_box[1] = self.top_right_corner_box_height
        
        return np.linalg.norm(pos_bottom_left_corner_cloth - pos_top_right_corner_box)
    
    def _compute_height_std(self):
        """计算布料高度的标准差，用于评估平整度"""
        pos = pyflex.get_positions().reshape(-1, 4)
        heights = pos[:, 1]
        return np.std(heights)
    
    def _compute_progress(self, edge_dist, corner_dist):
        """计算任务进度，基于边缘距离和角点距离的减少"""
        edge_progress = 1.0 - min(1.0, edge_dist / max(0.001, self.initial_edge_dist))
        corner_progress = 1.0 - min(1.0, corner_dist / max(0.001, self.initial_corner_dist))
        
        # 组合进度指标，优先考虑边缘对齐
        combined_progress = 0.7 * edge_progress + 0.3 * corner_progress
        
        # 平滑进度变化
        self.progress = 0.9 * self.progress + 0.1 * combined_progress
        return self.progress
    
    def compute_reward(self, action=None, obs=None, set_prev_reward=False, expert_action=None, rew_info=False):
        """增强的奖励函数，针对机器人折叠任务优化"""
        pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
        
        # 1. 边缘对齐奖励
        edge_dist = self._compute_edge_distance()
        edge_reward = -edge_dist
        
        # 2. 角点到目标的距离
        corner_dist = self._compute_corner_distance()
        corner_reward = -corner_dist
        
        # 3. 平整度奖励
        height_std = self._compute_height_std()
        flatness_reward = -height_std / max(0.001, self._compute_height_std())
        
        # 4. 计算任务进度
        progress = self._compute_progress(edge_dist, corner_dist)
        
        # 5. 阶段性调整权重
        # 初期更注重平整度和边缘对齐
        if progress < 0.3:
            edge_weight = 1.0
            corner_weight = 0.3
            flatness_weight = 0.5
        # 中期平衡关注各方面
        elif progress < 0.7:
            edge_weight = 1.0
            corner_weight = 0.7
            flatness_weight = 0.3
        # 后期更注重角点定位
        else:
            edge_weight = 0.5
            corner_weight = 1.0
            flatness_weight = 0.2
        
        # 组合奖励
        task_reward = (edge_weight * edge_reward + 
                      corner_weight * corner_reward + 
                      flatness_weight * flatness_reward)
        
        # 完成奖励 - 当接近完成时给予额外奖励
        completion_bonus = 0.0
        if edge_dist < 0.05 * self.initial_edge_dist and corner_dist < 0.05 * self.initial_corner_dist:
            completion_bonus = 5.0
        
        # 总任务奖励
        total_task_reward = task_reward + completion_bonus
        
        # 奖励变化加速学习
        reward_improvement = max(0, total_task_reward - self.last_task_reward)
        self.last_task_reward = total_task_reward
        
        # 模仿学习奖励（如启用）
        il_reward = 0
        if self.enable_rsi_ir and self.chosen_step < len(self.reference_next_state_info_ep):
            obs_key_points = self._get_obs_key_points()
            ref_dist = np.linalg.norm(self.reference_next_state_info_ep[self.chosen_step] - obs_key_points)
            il_reward = 1/(1 + np.exp(ref_dist))
            total_task_reward += il_reward
        
        # 添加改进奖励
        total_reward = total_task_reward + 0.3 * reward_improvement
        
        if rew_info:
            return total_reward, {
                'il_reward': il_reward if self.enable_rsi_ir else 0,
                'task_reward': task_reward,
                'edge_reward': edge_reward,
                'corner_reward': corner_reward,
                'flatness_reward': flatness_reward,
                'completion_bonus': completion_bonus,
                'progress': progress
            }
        
        return total_reward
    
    def _get_info(self):
        """扩展_get_info方法以包含更多性能指标"""
        info = super()._get_info()
        
        # 添加额外的性能指标
        edge_dist = self._compute_edge_distance()
        corner_dist = self._compute_corner_distance()
        height_std = self._compute_height_std()
        
        info.update({
            'edge_distance': edge_dist,
            'corner_distance': corner_dist,
            'height_std': height_std,
            'progress': self.progress,
        })
        
        return info