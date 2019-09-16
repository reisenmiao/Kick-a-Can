from controller import Supervisor, Node
from agent import Agent
import torch
import torch.nn.functional as F

# 初始化机器人
super_visor = Supervisor()
timestep = int(super_visor.getBasicTimeStep())


# 设置机器人的目标
target = torch.tensor([-0.5, 0.0, 0.0])


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
action_space = 2
state_space = 3
max_reward = torch.tensor([1.0])
agent = Agent(state_space, action_space)


# 机器人开始运行
position = torch.tensor(robot_node.getPosition())

# step counter
step_count = 0

while super_visor.step(timestep) != -1 :

    # take action
    action = agent.act(position)

    left_motor.setVelocity(action[0].item())
    right_motor.setVelocity(action[1].item())

    # get current state
    old_position = position
    position = torch.tensor(robot_node.getPosition())

    # calculate reward
    reward = max_reward - torch.pairwise_distance(position.unsqueeze(0), target.unsqueeze(0))

    # update value function
    agent.update(old_position, position, action, reward)













