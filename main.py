from Environment import Environment
from parse_utils import parse_dataset
from PolicyGradientAgent import PolicyGradientAgent

env = Environment()
X, y = parse_dataset(mode='train')

X_test, y_test = parse_dataset(mode='val')

agent = PolicyGradientAgent(pretrained=False)

agent.train(X, y)

agent.solve(X_test, y_test)
