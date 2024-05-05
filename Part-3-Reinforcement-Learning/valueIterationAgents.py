# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        """ If the state is terminal then value will be 0 and calculate the values using the transition probabilities for all actions, 
        and updating the value dictionary for each state and finally updating the self.values with the value dictionary.
        """
        for i in range(self.iterations):
            valueDict = util.Counter()
            totalStates = self.mdp.getStates()
            for currentState in totalStates:
                if self.mdp.isTerminal(currentState):     #Check for the terminal State
                    valueDict[currentState] = 0
                else:
                    valueofState = float("-inf")
                    totalActions = self.mdp.getPossibleActions(currentState)
                    for currentAction in totalActions:     #Iterate with all the actions possible
                        stateProbPair = self.mdp.getTransitionStatesAndProbs(currentState, currentAction)  #Get the Transition probabilities
                        totalValue = 0
                        for nextState, probValue in stateProbPair:
                            currentReward = self.mdp.getReward(currentState, currentAction, nextState)
                            totalValue = totalValue + probValue * (currentReward + (self.discount * self.values[nextState]))
                        valueofState = max(valueofState, totalValue)
                        valueDict[currentState] = valueofState
            self.values = valueDict

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        " Calculate the Value using the Transition Probabilities similar to approach used above"
        stateProbPair = self.mdp.getTransitionStatesAndProbs(state, action) #Get the Transition probabilities
        currentValue = 0
        for nextState, probValue in stateProbPair:   #Iterate through the state probabilities
            currentReward = self.mdp.getReward(state, action, nextState)
            currentValue = currentValue + probValue * (currentReward + (self.discount * self.values[nextState]))
        return currentValue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        "Computing action using QValues is nothing but returning the action associated with maximum QValue."
        bestPossibleAction = None
        currentValue = float("-inf")
        totalActions = self.mdp.getPossibleActions(state)     
        for currentAction in totalActions:       #Iterate through the total Actions
            qValue = self.computeQValueFromValues(state, currentAction)          #Get the QValue for the currentAction from the above function
            currentValue = max(currentValue, qValue)
            if(currentValue == qValue):
                bestPossibleAction = currentAction        
        return bestPossibleAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        """Here also we are computing the values for only one state in an iteration and value are computed in similar fashion."""
        totalStates = self.mdp.getStates()
        n = len(totalStates)
        for i in range(self.iterations):
            currentState = totalStates[i % n]
            if self.mdp.isTerminal(currentState):    #For Terminal state value is returned as 0.
                valueofState = 0
            else:
                valueofState = float("-inf")
                totalActions = self.mdp.getPossibleActions(currentState)    
                for currentAction in totalActions:
                    stateProbPair = self.mdp.getTransitionStatesAndProbs(currentState, currentAction)
                    totalValue = 0
                    for nextState, probValue in stateProbPair:
                        currentReward = self.mdp.getReward(currentState, currentAction, nextState)
                        totalValue = totalValue + probValue * (currentReward + (self.discount * self.values[nextState]))
                    valueofState = max(valueofState, totalValue)
            self.values[currentState] = valueofState             #Updating the self.values with the value computed.


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        "Initially find the predecessors of each state using the transition probabilities and populating the predecessors."
        predecessors = {}
        priortiyQueue = util.PriorityQueue()    
        totalStates = self.mdp.getStates()
        for currentState in totalStates:
            if not self.mdp.isTerminal(currentState):
                for currentAction in self.mdp.getPossibleActions(currentState):
                    stateProbPair = self.mdp.getTransitionStatesAndProbs(currentState, currentAction)
                    for nextState, probValue in stateProbPair:
                        if nextState in predecessors:
                            predecessors[nextState].add(currentState)
                        else:
                            predecessors[nextState] = {currentState}
        
        "Populate the priority queue with the max q values for each state."
        for currentState in totalStates:
            if not self.mdp.isTerminal(currentState):
                currentQValue = self.getMaxValue(currentState)
                diff = abs(self.values[currentState] - currentQValue)
                priortiyQueue.push(currentState, -diff)
        
        "Iterate over the iterations and for each state with highest priority compute the value and update it in priority queue again."
        for i in range(self.iterations):
            if priortiyQueue.isEmpty():
                break
            else:
                state = priortiyQueue.pop()
                if not self.mdp.isTerminal(state):
                    self.values[state] = self.getMaxValue(state)
                    for predState in predecessors[state]:
                        if not self.mdp.isTerminal(predState):
                            predQValue = self.getMaxValue(predState)
                            diff = abs(self.values[predState] - predQValue)
                            if diff > self.theta:
                                priortiyQueue.update(predState, -diff)

    "This function is used as helper function to find the maximum Q value for a state with possible actions."
    def getMaxValue(self, state):
        stateValue = float("-inf")
        for action in self.mdp.getPossibleActions(state):
            stateValue = max(stateValue, self.computeQValueFromValues(state, action))
        return stateValue