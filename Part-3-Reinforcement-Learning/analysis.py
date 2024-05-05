# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    "In order to cross the bridge the agent should go only in the east direction. So, it should follow the direction without any noise."
    answerDiscount = 0.9
    answerNoise = 0.001
    return answerDiscount, answerNoise

def question3a():
    """In order to take small reward and risking the clif it should have less noise and discount factor should be less for the agent 
    so that agent goes to terminal state faster, and also living reward should be negative for the agent to end the game quickly."""
    answerDiscount = 0.3
    answerNoise = 0
    answerLivingReward = -1.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    """In order to take small reward and avoiding the clif it should have some noise so that it won't take the path,
    As the route is longer we take discount factor some what more than the above case for longer survival,
    and also living reward should be negative for the agent to end the game quickly."""
    answerDiscount = 0.5
    answerNoise = 0.2
    answerLivingReward = -2.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    """ In order to take cliff route again the noise should be 0 and here living reward should be positive and comparable to small reward,
    so that it won't take the small reward terminal state and goes for the big reward terminal state.
    """
    answerDiscount = 0.4
    answerNoise = 0
    answerLivingReward = 0.5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    """In order to take high reward terminal state and avoiding cliff we need some noise and high discount factor for survival
    and living reward should be small negative value for us to reach terminal states.
    """
    answerDiscount = 0.8
    answerNoise = 0.2
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    """In order for the agent to be avoiding terminal states and the cliff we need to have the living reward as high as the high terminal state."""
    answerDiscount = 1.0
    answerNoise = 0.0
    answerLivingReward = 10.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    """This case is not possible so we tend to return NOT POSSIBLE."""
    answerEpsilon = None
    answerLearningRate = None
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
