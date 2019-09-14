from controller import Supervisor, Node
from agent import Random
import torch
import torch.nn.functional as F

# 初始化机器人
super_visor = Supervisor()
timestep = int(super_visor.getBasicTimeStep())


# 设置机器人的目标
target = torch.tensor([-0.5, 0.0, 0.0]).view(1, -1)


# 获取机器人的node
root = super_visor.getRoot()
children = root.getField("children")
n = children.getCount()
robot_node = children.getMFNode(n - 1)


# 初始化左右引擎
left_motor = super_visor.getMotor("left wheel motor")
right_motor = super_visor.getMotor("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)


# 设置动作和agent
action_space = [(10.0, 10.0), (-10.0, 10.0), (10.0, -10.0), (-10.0, -10.0), (0.0, 0.0)]
agent = Random(action_space, 3)


# 机器人开始运行
position = torch.tensor(robot_node.getPosition()).view(1, -1)

# step counter
step_count = 0

while super_visor.step(timestep) != -1 :

    # take action
    action_index = agent.act(position)

    left_motor.setVelocity(action_space[action_index.item()][0])
    right_motor.setVelocity(action_space[action_index.item()][1])

    # get current state
    old_position = position
    position = torch.tensor(robot_node.getPosition()).view(1, -1)

    # calculate reward
    reward = torch.tensor([1 / F.pairwise_distance(position, target)])

    # update value function
    agent.update(old_position, position, action_index, reward)













