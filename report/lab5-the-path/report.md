## Lab5 路径规划

### 复现命令：

启动 CoppeliaSim 场景 `Robot3 - velocity.ttt`，确保 ZMQ 接口开启，然后在命令行运行以下命令：

```pwsh
uv.exe run myarm velocity --traj square --center 0.2 0.0 0.2
uv.exe run myarm velocity --traj circle --center 0.2 0.0 0.2
uv.exe run myarm velocity --traj cone
```

#### 效果视频：

【小小机械臂】 https://www.bilibili.com/video/BV17UycByE7Y/?share_source=copy_web&vd_source=8b8f514c84fa7c5c5501351bb43b2176

### 过程介绍

思路上先在 `traj square|circle|cone` 里构造 `TrajState`，把位置、速度、姿态和角速度都准备好：方形和圆形只在 X-Y 平面里规划顶点或极坐标，自动把米级参数换算成毫米并在首尾 12% 的轨迹上做速度缓启动/缓停；锥面轨迹则根据世界坐标系下的旋转轴生成一组正交基，再用半顶角决定末端的 z 轴方向，保证姿态沿轴匀速进动。CLI 把 `--center` 参数乘以 1000 交给 `build_traj_fn`，这样一套命令就能在 CoppeliaSim 里复用。

运行流程是：先连到 ZMQ 接口获取关节和末端句柄，之后主循环以 `dt=20ms` 周期读取关节位置、做一次正解拿到当前位姿，与轨迹状态做位置/姿态误差，通过两个比例环（`kp_pos`, `kp_rot`）组合出末端期望扭量，再用阻尼伪逆 `damped_pinv_step` 解出关节速度，最后用 `clip_joint_velocities` 限制在 2rad/s 内并推送给模拟器。若 `--draw`，顺便把 tip 点位塞进画线 buffer 便于观察路径。整个过程就是“离线生成轨迹 → 在线 Jacobian 追踪 → 可视化确认”，调参时只需要调整速度或增益就能兼顾稳定性与响应速度。

### 目前问题

初始没有设置关节位置，靠 Jacobian 追踪到初始位置。不知道为什么 `sim.setJointPosition` 和 `sim.setJointTargetPosition` 都是 no-op，可能是只要设置成速度模式就不能用了（？），报错也没。

三个轨迹的位置都是精心挑选的，尤其那个cone，反复尝试后，中心定在[0.100, 0.100, 0.500]，轴线是世界z轴，是为了避免机械臂在运动过程中碰撞到桌面（link2往下转过头）和自身（吸头撞在自己连杆上）。剩下的square和circle在水平面运动，没有那么诡异的姿态变化，相对好挑。直接定在[0.200, 0.000, 0.200]，这是一个比较近而且有一定高度的位置。

前两个的速度`--speed`参数都设成0.05m/s，第三个锥面轨迹的角速度`--ang-speed`设成0.8rad/s，高了抖动非常厉害，观赏性不高。