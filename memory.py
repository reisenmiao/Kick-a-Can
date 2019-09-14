from collections import namedtuple

transition = namedtuple('Transition', ('state', 'action', 'next_sate', 'reward'))
