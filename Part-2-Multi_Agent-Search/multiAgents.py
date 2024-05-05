# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
from math import inf
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newCapsule = successorGameState.getCapsules()

        "*** YOUR CODE HERE ***"
        totalScore = successorGameState.getScore()
        reward = 10    # Let's Define reward as 10 this suits perfectly well for the logic written below.
        closeProximity = 3 # This is like a minimum close proximity the pacman can be near to ghost. 

        "Criteria -1: Find the closest food in the new position and get the score according the distance it is from the pacman."

        distanceFromClosestFood = inf
        for food in newFood.asList():
            distanceFromClosestFood = min(distanceFromClosestFood, manhattanDistance(newPos, food))
        totalScore += reward/distanceFromClosestFood

        """Criteria-2: Find the closest ghost distance from that new position and if it is less than the close proximity 
        then we negate our score,so that pacman wont take that route. 
        Also we calculate number of ghosts which are in scaredTime and we add them to the total score."""
        distanceFromClosestGhost = inf
        numberofScaredGhosts = 0
        for ghost in newGhostStates:
            ghostPosition = ghost.getPosition()
            if ghost.scaredTimer is 0:
                distanceFromClosestGhost = min(distanceFromClosestGhost, manhattanDistance(newPos, ghostPosition))
            else:
                numberofScaredGhosts +=1

        if distanceFromClosestGhost <= closeProximity:
            totalScore -= reward*(reward/(distanceFromClosestGhost+1))
        
        totalScore += 3*reward*numberofScaredGhosts

        "Criteria-3: If the position we are going to has food then we can add that to score."
        "If it is a capsule then we can add more reward as it will make ghost scared for some time"
        if newPos in newFood:
            totalScore += reward
        
        if newPos in newCapsule:
            totalScore += 5*reward

        return totalScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        resultantActionValue = self.minMaxValueFunction(gameState, 0, 0)
        return resultantActionValue[1]
    
    def minMaxValueFunction(self, state, index, depth):

        "Checking for any Terminal States which gives us the boolean value of whether the state is terminal or not."
        if state.isWin() or state.isLose() or len(state.getLegalActions(index)) == 0 or depth >= self.depth:
            return self.evaluationFunction(state), None  # If is is terminal state then return its current value.
        else:
            movesLeft = state.getLegalActions(index) # Find the Legal moves left
            "For Max agent which is Pacman we recursively call for the values and get the maximum value of them all as it is a max-node."
            if index == 0:
                resultValue = -inf
                resultMove = None
                for move in movesLeft:
                    succ = state.generateSuccessor(index, move)
                    succIndex = index+1
                    succDepth = depth
                    if succIndex == state.getNumAgents():
                        succDepth +=1
                        succIndex = 0
                    tempValue = self.minMaxValueFunction(succ, succIndex, succDepth)[0]  #Find the value of the successor
                    resultValue = max(resultValue, tempValue) # Update it only if it is maximum among the computed values.
                    if resultValue == tempValue:
                        resultMove = move
            else:
                "For Min agent such as ghosts we recursively call for the values and get the minimum value of them all as it is a min-node."
                resultValue = inf
                resultMove = None
                for move in movesLeft:
                    succ = state.generateSuccessor(index, move)
                    succIndex = index+1
                    succDepth = depth

                    if succIndex == state.getNumAgents():
                        succDepth +=1
                        succIndex = 0
                    
                    tempValue = self.minMaxValueFunction(succ, succIndex, succDepth)[0]  #Find the value of the successor
                    resultValue = min(resultValue, tempValue) # Update it only if it is minimum among the computed values.

                    if resultValue == tempValue:
                        resultMove = move

        return resultValue, resultMove
            

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -inf
        beta = inf
        result = self.alphaBetaPruningFunction(gameState, 0, 0, alpha, beta)
        return result[1]

    def alphaBetaPruningFunction(self, state, index, depth, alphaValue, betaValue):

        if state.isWin() or state.isLose() or len(state.getLegalActions(index)) == 0 or depth >= self.depth:  
            return self.evaluationFunction(state), None   # If is is terminal state then return its current value.
        else:
            movesLeft = state.getLegalActions(index)
            if index == 0:
                """For Max agents which is Pacman we recursively call for the values and get the maximum value of the successor,
                Additionally we also have to maintain alpha and beta values which will reduce our computations.
                If the value is greater than the beta value in the computation tree then we return the value from that point
                and ignore all the subtrees below."""  
                resultValue = -inf
                resultMove = None
                for move in movesLeft:
                    succ = state.generateSuccessor(index, move)
                    succIndex = index+1
                    succDepth = depth

                    if succIndex == state.getNumAgents():
                        succDepth +=1
                        succIndex = 0
                    
                    tempValue = self.alphaBetaPruningFunction(succ, succIndex, succDepth, alphaValue, betaValue)[0]
                    resultValue = max(resultValue, tempValue)
                    alphaValue = max(alphaValue, resultValue)

                    if resultValue > betaValue:    #If the value is greater than the beta value then return it and wont check for any other nodes in that sub tree.
                        return resultValue, move
                    elif resultValue == tempValue:
                        resultMove = move
            else:
                """For Min agents which are Ghosts which recursively call for the values and get the minimum value of the successor,
                Additionally we also have to maintain alpha and beta values which will reduce our computations.
                If the value is less than the alpha value in the computation tree then we return the value from that point
                and ignore all the subtrees below."""
                resultValue = inf
                resultMove = None
                
                for move in movesLeft:
                    succ = state.generateSuccessor(index, move)
                    succIndex = index+1
                    succDepth = depth

                    if succIndex == state.getNumAgents():
                        succDepth +=1
                        succIndex = 0
                    
                    tempValue = self.alphaBetaPruningFunction(succ, succIndex, succDepth, alphaValue, betaValue)[0]
                    resultValue = min(resultValue, tempValue)
                    betaValue = min(betaValue, resultValue)

                    if resultValue < alphaValue:
                        return resultValue, move
                    elif resultValue == tempValue:
                        resultMove = move

        return resultValue, resultMove

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    We are trying to evaluate the score based on these parameters:
       -- 1. closest Food Position: We will add score for closestFood from the current state in terms of its closeness to the state
       -- 2. closest Capsule Position: We will add score for closest Capsule from the current state in terms of its closeness to the state
                                    Reward is multiplied by 2 as capsule is more effective than normal food
       -- 3. closest Ghost Position: We will subtract score for closest Ghost from the current state in terms of its closeness to the state.
       -- 4. number of scared Ghosts: We will add the score for number of scared ghosts present at that time.
       -- 5. Food Left over: We need to subtract the score as per the food left over in the game.
       -- 6. Capsules Left over: We need to subtract the score as per the capsules left over in the game, 
                              Need to multiply by 2 as capsules are more valuable.

    We are defining a reward as 10 (some random number)
    """
    "*** YOUR CODE HERE ***"
    currentPosition = currentGameState.getPacmanPosition()
    foodLocation = currentGameState.getFood().asList()
    capsuleLocation = currentGameState.getCapsules()
    ghostPosition = currentGameState.getGhostPositions()
    ghostStates = currentGameState.getGhostStates()
    foodLeft = len(foodLocation)
    capsulesLeft = len(capsuleLocation)
    score = currentGameState.getScore()
    reward = 10

    closestFood = inf
    closestCapsule = inf
    closestGhost = inf
    scaredGhosts = 0

    if currentGameState.isWin():
        return inf
    elif currentGameState.isLose():
        return -inf

    for food in foodLocation:
        closestFood = min(closestFood, manhattanDistance(currentPosition, food))
    
    score += reward/(closestFood+1)

    for capsule in capsuleLocation:
        closestCapsule = min(closestCapsule, manhattanDistance(currentPosition, capsule))

    score += 2*(reward/(closestCapsule+1))

    for ghost in ghostStates:
        ghostPosition = ghost.getPosition()
        if(ghost.scaredTimer is 0):
            closestGhost = min(closestGhost, manhattanDistance(currentPosition, ghostPosition))
        else:
            scaredGhosts += 1
    
    score -= 5*(reward/(closestGhost+1))
    score += 3*(reward * scaredGhosts)
    score -= 2*(reward*capsulesLeft)
    score -= (reward/2)*foodLeft
    return score


# Abbreviation
better = betterEvaluationFunction
