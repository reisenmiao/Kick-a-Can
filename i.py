from controller import Supervisor, Node
from agent import Agent
import torch
import torch.nn.functional as F

# 初始化机器人
super_visor = Supervisor()
timestep = int(super_visor.getBasicTimeStep())


# 设置机器人的目标
target = torch.tensor([-0.5, 0.0, 0.0]).unsqueeze(0)


# 获取机器人的node和field
root = super_visor.getRoot()
children = root.getField("children")
n = children.getCount()
robot_node = children.getMFNode(n - 1)

translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

# 初始化左右引擎
left_motor = super_visor.getMotor("left wheel motor")
right_motor = super_visor.getMotor("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)


# 设置state, action和agent
action_space = 2
state_space = 4
max_reward = torch.tensor([2.0])
agent = Agent(state_space, action_space)


# 机器人开始运行
robot_name = robot_node.getDef()
print("Robot {} starts!".format(robot_name))

position = torch.tensor(translation_field.getSFVec3f()).unsqueeze(0)
orientation = torch.tensor([rotation_field.getSFRotation()[3]]).unsqueeze(0)

state = torch.cat((position, orientation), dim=1)

# step counter
step_count = 0

while super_visor.step(timestep) != -1 :

    # take action
    action = agent.act(state)

    left_motor.setVelocity(action[0][0].item())
    right_motor.setVelocity(action[0][1].item())

    # get current state
    position = torch.tensor(translation_field.getSFVec3f()).unsqueeze(0)
    orientation = torch.tensor([rotation_field.getSFRotation()[3]]).unsqueeze(0)

    old_state = state
    state = torch.cat((position, orientation), dim=1)

    # calculate r
    # eward
    reward = max_reward - torch.pairwise_distance(position, target)

    # memorize current transition
    agent.memory.push(old_state, action, state, reward)

    # perform optimization
    agent.update()













