# search.py
# ---------
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
"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from util import Stack, Queue, PriorityQueue


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """
    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def genericSearch(problem, fringe):

    visited = set()
    totalPath = list()
    fringe.push((problem.getStartState(), list(), 0))
    while not fringe.isEmpty():
        currentState = fringe.pop()
        if problem.isGoalState(currentState[0]) == True:
            return currentState[1]
        if currentState[0] not in visited:
            for childNode, action, childCost in problem.getSuccessors(currentState[0]):
                    totalPath = currentState[1].copy()
                    totalPath.append(action)
                    totalCost = currentState[2] + childCost
                    fringe.push((childNode, totalPath, totalCost))
        visited.add(currentState[0])

    return None


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    stack = Stack()
    # We are using Stack in case of Depth First Search and the nodes are popped as LIFO order,
    # Note: The code was developed similar to the generic search method given above.
    startingState = problem.getStartState()
    pathFollowed = []
    startState = (startingState, pathFollowed)
    stack.push(startState)
    cloList = []
    while not stack.isEmpty():
        currentState, currentPath = stack.pop()
        if problem.isGoalState(currentState):
            return currentPath
        if currentState not in cloList:
            cloList.append(currentState)
            childData = problem.getSuccessors(currentState)
            for state, path, cost in childData:
                updatedPath = currentPath + [path]
                childState = (state, updatedPath)
                stack.push(childState)
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = Queue()
    # We are using Queue in case of Breadth First Search and the nodes are popped as FIFO order.
    # Note: The code was developed similar to the generic search method given above.
    startingState = problem.getStartState()
    pathFollowed = []
    startState = (startingState, pathFollowed)
    queue.push(startState)
    visited = []
    while not queue.isEmpty():
        currentState, currentPath = queue.pop()
        if problem.isGoalState(currentState):
            return currentPath
        if currentState not in visited:
            visited.append(currentState)
            childData = problem.getSuccessors(currentState)
            for state, path, cost in childData:
                updatedPath = currentPath + [path]
                childState = (state, updatedPath)
                queue.push(childState)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    priorityQueue = PriorityQueue()
    """We are using Priority Queue in case of Uniform Cost Search and priority is the g-value of the states,
    which can be computed using getCostOfActions Function.
    # Note: The code was developed similar to the generic search method given above.
    """
    startingState = problem.getStartState()
    pathFollowed = []
    startState = (startingState, pathFollowed)
    priorityQueue.push(startState, 0)
    visited = []
    while not priorityQueue.isEmpty():
        currentState, currentPath = priorityQueue.pop()
        if problem.isGoalState(currentState):
            return currentPath
        if currentState not in visited:
            visited.append(currentState)
            childData = problem.getSuccessors(currentState)
            for state, path, cost in childData:
                if state not in visited:
                    updatedPath = currentPath + [path]
                    costTillNow = problem.getCostOfActions(updatedPath)
                    child = (state, updatedPath)
                    priorityQueue.push(child, costTillNow)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    priorityQueue = PriorityQueue()
    """Here we are again using priority Queue with a slight modification, the priority is the f-value of the states,
    which is the sum of the g-value(which can be computed using getCostOfActions function)
    and h-value (which can be computed using heuristic which is being passed to the function)
    f-value = g-value + h-value.
    # Note: The code was developed similar to the generic search method given above.
    """
    startingState = problem.getStartState()
    pathFollowed = []
    startState = (startingState, pathFollowed)
    priorityQueue.push(startState, 0)
    visited = []
    while not priorityQueue.isEmpty():
        currentState, currentPath = priorityQueue.pop()
        if problem.isGoalState(currentState):
            return currentPath
        if currentState not in visited:
            visited.append(currentState)
            childData = problem.getSuccessors(currentState)
            for state, path, cost in childData:
                if state not in visited:
                    updatedPath = currentPath + [path]
                    costTillNow = problem.getCostOfActions(updatedPath) + heuristic(state, problem)
                    child = (state, updatedPath)
                    priorityQueue.push(child, costTillNow)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
