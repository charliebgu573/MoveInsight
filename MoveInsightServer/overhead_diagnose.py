# 需要判断的7个技术指标如下：
#   1. **shoulder_abduction**: 不要掉肘；
#   2. **elbow_flexion**: 肘关节收紧；
#   3. **elbow_lower**: 持拍手的肘部低于非持拍手的肘部；
#   4. **foot_direction_aligned**: 双脚的脚尖方向是否一致；
#   5. **proximal_to_distal_sequence**: 挥拍阶段上肢发力的顺序是否正确；
#   6. **hip_forward_shift**: 挥拍阶段髋部前移；
#   7. **trunk_rotation_completed**: 挥拍阶段躯干转体是否充分。

import numpy as np
def evaluate_swing_rules(keypoints, dominant_side='Right'):
    """
    KEYPOINTS 输入说明：

    本系统使用的关键点数据格式为 Python 字典（dict），名为 `keypoints`，其结构如下：

    keypoints = {
        'RightShoulder': np.ndarray of shape (T, 2),
        'LeftShoulder':  np.ndarray of shape (T, 2),
        'RightElbow':    np.ndarray of shape (T, 2),
        'LeftElbow':     np.ndarray of shape (T, 2),
        'RightWrist':    np.ndarray of shape (T, 2),
        'RightHand':     np.ndarray of shape (T, 2),  # 可选
        'RightHip':      np.ndarray of shape (T, 2),
        'LeftHip':       np.ndarray of shape (T, 2),
        'RightHeel':     np.ndarray of shape (T, 2),
        'RightToe':      np.ndarray of shape (T, 2),
        'LeftHeel':      np.ndarray of shape (T, 2),
        'LeftToe':       np.ndarray of shape (T, 2)
    }

    说明：
    - 每个 key 是关键点的英文全称；
    - 每个 value 是一个 numpy 数组，形状为 (T, 2)，其中：
        - T 表示视频的帧数；
        - 每一行是该帧的 (X, Y) 坐标；
        - 坐标单位可为像素或归一化值，但整组数据应保持一致。

    数据来源：
    - 可通过 Apple Vision, MediaPipe, OpenPose 等人体姿态识别工具生成；
    - 你应根据模型输出映射 keypoint 编号，统一重命名为上述英文标准名；
    - 可在中间添加字典映射，例如：
        joint_map = {0: 'RightShoulder', 1: 'RightElbow', ...}
    """
    result = {}
    D = dominant_side
    ND = 'Left' if D == 'Right' else 'Right'
    T = keypoints[f'{D}Shoulder'].shape[0]

    def angle_between(v1, v2):
        unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)
        unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)
        dot = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    def joint_angle(A, B, C):
        v1 = A - B
        v2 = C - B
        return angle_between(v1, v2)

    # Direction-based phase split using elbow movement
    elbow_x_vel = np.zeros(T)
    elbow_x_vel[1:] = keypoints[f'{D}Elbow'][1:, 0] - keypoints[f'{D}Elbow'][:-1, 0]

    # Apply smoothing to reduce noise (elbow is more active than shoulder, so medium window)
    window_size = min(5, T - 1)
    if window_size > 2:
        smoothed_vel = np.convolve(elbow_x_vel, np.ones(window_size) / window_size, mode='valid')
        padding = np.zeros(T - len(smoothed_vel))
        smoothed_vel = np.concatenate((padding, smoothed_vel))
    else:
        smoothed_vel = elbow_x_vel

    # Find the point where velocity changes from negative to positive
    # This indicates the elbow stops moving backward and starts moving forward
    swing_start = 0
    for t in range(1, T):
        if smoothed_vel[t - 1] < 0 and smoothed_vel[t] > 0:
            swing_start = t
            break

    # Fallback if no clear turning point found
    if swing_start == 0:
        vel_threshold = np.max(np.abs(smoothed_vel)) * 0.15
        for t in range(1, T):
            if smoothed_vel[t] > vel_threshold:
                swing_start = t
                break

        if swing_start == 0:
            swing_start = T // 2

    swing_end = T - 1

    # Keep the degenerate case check
    if swing_start >= swing_end:
        result.update({
            'shoulder_abduction': False,
            'elbow_flexion': False,
            'elbow_lower': False,
            'foot_direction_aligned': False,
            'proximal_to_distal_sequence': False,
            'hip_forward_shift': False,
            'trunk_rotation_completed': False
        })
        return result

    # 引拍期
    shoulder_angles = []
    elbow_angles = []
    for t in range(swing_start):
        shoulder_angles.append(joint_angle(keypoints[f'{D}Hip'][t], keypoints[f'{D}Shoulder'][t], keypoints[f'{D}Elbow'][t]))
        elbow_angles.append(joint_angle(keypoints[f'{D}Shoulder'][t], keypoints[f'{D}Elbow'][t], keypoints[f'{D}Wrist'][t]))

    result['shoulder_abduction'] = np.any((np.array(shoulder_angles) >= 60) & (np.array(shoulder_angles) <= 90))
    result['elbow_flexion'] = np.any(np.array(elbow_angles) < 90)
    result['elbow_lower'] = np.any(keypoints[f'{D}Elbow'][:swing_start, 1] > keypoints[f'{ND}Elbow'][:swing_start, 1])

    # foot direction
    v_r = keypoints['RightToe'][:swing_start] - keypoints['RightHeel'][:swing_start]
    v_l = keypoints['LeftToe'][:swing_start] - keypoints['LeftHeel'][:swing_start]
    foot_angles = [angle_between(vr, vl) for vr, vl in zip(v_r, v_l)]
    result['foot_direction_aligned'] = np.any(np.array(foot_angles) < 30)

    # 挥拍期
    def get_angle_series(A, B, C):
        return np.array([joint_angle(A[t], B[t], C[t]) for t in range(T)])

    shoulder_seq = get_angle_series(keypoints[f'{D}Hip'], keypoints[f'{D}Shoulder'], keypoints[f'{D}Elbow'])
    elbow_seq = get_angle_series(keypoints[f'{D}Shoulder'], keypoints[f'{D}Elbow'], keypoints[f'{D}Wrist'])
    wrist_seq = get_angle_series(keypoints[f'{D}Elbow'], keypoints[f'{D}Wrist'],
                                 keypoints[f'{D}Hand'] if f'{D}Hand' in keypoints else keypoints[f'{D}Wrist'])

    vel_shoulder = np.abs(np.diff(shoulder_seq))
    vel_elbow = np.abs(np.diff(elbow_seq))
    vel_wrist = np.abs(np.diff(wrist_seq))

    result['proximal_to_distal_sequence'] = False
    if swing_start < len(vel_shoulder):
        t_s = np.argmax(vel_shoulder[swing_start:]) + swing_start
        t_e = np.argmax(vel_elbow[swing_start:]) + swing_start
        t_w = np.argmax(vel_wrist[swing_start:]) + swing_start
        result['proximal_to_distal_sequence'] = t_s < t_e #手腕的识别不准先去掉了，只看了肩和肘的

    hip_center_x = 0.5 * (keypoints[f'{D}Hip'][:, 0] + keypoints[f'{ND}Hip'][:, 0])
    hip_range = np.max(hip_center_x) - np.min(hip_center_x)
    result['hip_forward_shift'] = hip_range >= 0.10

    dist_init = np.linalg.norm(keypoints[f'{D}Shoulder'][swing_start] - keypoints[f'{ND}Shoulder'][swing_start])
    dist_min = np.min([
        np.linalg.norm(keypoints[f'{D}Shoulder'][t] - keypoints[f'{ND}Shoulder'][t])
        for t in range(swing_start, swing_end + 1)
    ])
    result['trunk_rotation_completed'] = (dist_init - dist_min) / (dist_init + 1e-6) >= 0.5

    return result

def align_keypoints_with_interpolation(joint_data, frame_count):
    keypoints = {}

    for name, points in joint_data.items():
        pts = np.array(points)  # shape: (n, 2)
        n = len(pts)

        if n == frame_count:
            keypoints[name] = pts
        elif n >= frame_count - 2:
            # ⚠️ 允许最多缺 1-2 帧，尝试修复

            # 创建一个 shape=(frame_count, 2) 的空数组
            filled = np.full((frame_count, 2), np.nan)

            # 填入已知帧（假设 joint_data 是顺序连续 append 的）
            valid_indices = np.linspace(0, frame_count - 1, n, dtype=int)
            filled[valid_indices] = pts

            # 找出缺失帧，用线性插值补齐
            for dim in range(2):  # 对X和Y分别插值
                valid = ~np.isnan(filled[:, dim])
                if np.sum(valid) < 2:
                    print(f"❌ 无法插值 {name}，有效点太少")
                    break
                filled[:, dim] = np.interp(
                    np.arange(frame_count),
                    np.where(valid)[0],
                    filled[valid, dim]
                )

            keypoints[name] = filled

        else:
            print(f"⚠️ {name} 缺失太多（{n}/{frame_count}），丢弃该关键点")

    return keypoints

# #########################################测试函数功能############################################
T = 100
def gen_mock_data(x_shift=0.0):
    # Linear motion with optional shift
    return np.stack([np.linspace(0, 1, T) + x_shift, np.linspace(0.5, 0.5, T)], axis=1)

keypoints = {
    'RightShoulder': gen_mock_data(0.1),
    'LeftShoulder': gen_mock_data(-0.1),
    'RightElbow': gen_mock_data(0.2),
    'LeftElbow': gen_mock_data(-0.2),
    'RightWrist': gen_mock_data(0.3),
    'RightHand': gen_mock_data(0.35),
    'RightHip': gen_mock_data(0.0),
    'LeftHip': gen_mock_data(-0.05),
    'RightToe': gen_mock_data(0.2),
    'RightHeel': gen_mock_data(0.1),
    'LeftToe': gen_mock_data(-0.2),
    'LeftHeel': gen_mock_data(-0.1),
}

# Evaluate rules
results = evaluate_swing_rules(keypoints, dominant_side='Right')
print(results)
