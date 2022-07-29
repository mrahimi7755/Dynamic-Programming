import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import main_agent
import ten_arm_env
import test_env

def argmax(q_values):
  """
    Takes in a list of q_values and returns the index of the item 
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    top_value = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i]>top_value:
            top_value = q_values[i]
            ties = [1]
            ties[0] = i
        elif q_values[i]==top_value:
            ties.append(i)  
        #         raise NotImplementedError()
    return np.random.choice(ties)

class GreedyAgent(main_agent.Agent):
# Update Q values Hint: Look at the algorithm in section 2.4 of the textbook.
# increment the counter in self.arm_count for the action from the previous time step
# update the step size using self.arm_count
# update self.q_values for the action from the previous time step

    def agent_step(self, reward, observation=None):
        """
        Takes one step for the agent. It takes in a reward and observation and 
        returns the action the agent chooses at that time step.

        Arguments:
        reward -- float, the reward the agent recieved from the environment after taking the last action.
        observation -- float, the observed state the agent is in. Do not worry about this as you will not use it
                              until future lessons
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """
        self.arm_count[self.last_action]+=1
        self.q_values[self.last_action] = self.q_values[self.last_action]+(reward-self.q_values[self.last_action])/self.arm_count[self.last_action]
        current_action=argmax(self.q_values)
        self.last_action = current_action
        return current_action
        
class EpsilonGreedyAgent(main_agent.Agent):
    def agent_step(self, reward, observation):
        """
        Takes one step for the agent. It takes in a reward and observation and 
        returns the action the agent chooses at that time step.

        Arguments:
        reward -- float, the reward the agent recieved from the environment after taking the last action.
        observation -- float, the observed state the agent is in. Do not worry about this as you will not use it
                              until future lessons
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """

        ### Useful Class Variables ###
        # self.q_values : An array with what the agent believes each of the values of the arm are.
        # self.arm_count : An array with a count of the number of times each arm has been pulled.
        # self.last_action : The action that the agent took on the previous time step
        # self.epsilon : The probability an epsilon greedy agent will explore (ranges between 0 and 1)
        #######################

        # Update Q values - this should be the same update as your greedy agent above
        # YOUR CODE HERE
        self.arm_count[self.last_action]+=1
        self.q_values[self.last_action] = self.q_values[self.last_action]+(reward-self.q_values[self.last_action])/self.arm_count[self.last_action]        
        # Choose action using epsilon greedy
        # Randomly choose a number between 0 and 1 and see if it's less than self.epsilon
        # (hint: look at np.random.random()). If it is, set current_action to a random action.
        # otherwise choose current_action greedily as you did above.
        # YOUR CODE HERE
        if np.random.random()<self.epsilon:
            current_action = np.random.choice(range(self.num_actions))
        else:
            current_action = argmax(self.q_values)
        self.last_action = current_action
        return current_action

class EpsilonGreedyAgentConstantStepsize(main_agent.Agent):
    def agent_step(self, reward, observation):
        """
        Takes one step for the agent. It takes in a reward and observation and 
        returns the action the agent chooses at that time step.

        Arguments:
        reward -- float, the reward the agent recieved from the environment after taking the last action.
        observation -- float, the observed state the agent is in. Do not worry about this as you will not use it
                              until future lessons
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """

        ### Useful Class Variables ###
        # self.q_values : An array with what the agent believes each of the values of the arm are.
        # self.arm_count : An array with a count of the number of times each arm has been pulled.
        # self.last_action : An int of the action that the agent took on the previous time step.
        # self.step_size : A float which is the current step size for the agent.
        # self.epsilon : The probability an epsilon greedy agent will explore (ranges between 0 and 1)
        #######################

        # Update q_values for action taken at previous time step 
        # using self.step_size intead of using self.arm_count
        self.arm_count[self.last_action]+=1
        self.q_values[self.last_action] = self.q_values[self.last_action]+(reward-self.q_values[self.last_action])*self.step_size        
        if np.random.random()<self.epsilon:
            current_action = np.random.choice(range(self.num_actions))
        else:
            current_action=argmax(self.q_values)
        self.last_action = current_action
        return current_action
