# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import copy

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.values = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return float(self.values[(state, action)])
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        return 0.0 + max([self.getQValue(state,action) for action in self.getLegalActions(state)])

        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if actions:
            values = [self.getQValue(state,action) for action in actions]
            return actions[values.index(max(values))]
        else:
            return None

        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        if not util.flipCoin(self.epsilon):
            action = self.computeActionFromQValues(state)
        else:
            action = random.choice(self.getLegalActions(state))

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        nextActions = self.getLegalActions(nextState)
        if nextActions:
            self.values[(state,action)] = (1.0 - self.alpha)*self.getQValue(state,action) + \
                                      self.alpha*(reward + self.discount*max([self.getQValue(nextState, nextAction) \
                                                                              for nextAction in nextActions]))
        else:
            self.values[(state, action)] = (1.0 - self.alpha) * self.getQValue(state, action) + self.alpha * reward

        return

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        f = self.featExtractor.getFeatures(state, action)
        fkeys = f.keys()

        qvalue = 0.0

        for k in fkeys:
            qvalue = qvalue + (self.getWeights()[k] * f[k])

        return qvalue

        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        f = self.featExtractor.getFeatures(state, action)
        fkeys = f.keys()
        temp = util.Counter()

        for k in fkeys:
            nextActions = self.getLegalActions(nextState)
            if nextActions:
                diff = (float(reward) + (self.discount *
                        max([self.getQValue(nextState, nextAction) for nextAction in nextActions]))) \
                        - self.getQValue(state, action)

            else:
                diff = (float(reward) - self.getQValue(state, action))

            temp[k] = self.weights[k] + (self.alpha * diff * float(f[k]))

        for k in fkeys:
            self.weights[k] = float(temp[k])


        return

        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

'''
The Sarsa lambda agent keeps an eligibility trace, updating the q values
for the last y states at a certain 
'''
class SarsaLambdaAgent(QLearningAgent):
    def __init__(self, y=.8, trace_length = None, **args):
        self.y = y
        self.eligibility = util.Counter()
        self.predicted_next_action = None
        self.prediction_made = False
        QLearningAgent.__init__(self, **args)
    
    #getAction() must call doAction() for sim to work properly,
    #But we also need to know the agent's next action ahead of time
    #to update properly. solution: decideAction decides.
    #which action will be taken in the future so that eligibility 
    #and values can be updated correctly
    def decideAction(self, state):

        # Pick Action
        if self.prediction_made:
            action = self.predicted_next_action
        else:
            legalActions = self.getLegalActions(state)
            action = None
            if legalActions:
                if util.flipCoin(self.epsilon):
                    action = random.choice(self.getLegalActions(state))
                else:
                    action = self.computeActionFromQValues(state)
                self.predicted_next_action = action
                self.prediction_made = True
        return action

    #returns the predicted action if it has not been taken
    #yet otherwise return next future action
    def getAction(self, state):
        if self.prediction_made:
            a = self.predicted_next_action
        else:
            a = self.decideAction(state)
        self.doAction(state, a)
        return a

    def doAction(self, state, action):
        self.prediction_made = False
        QLearningAgent.doAction(self, state, action)

    #update values and based on the previous values and current 
    #eligibulity trace.  The current (state,action) is updated
    #in eligibility to keep a record to how often an action has
    #has been taken
    def update(self, state, action, nextState, reward):
        nextAction = self.decideAction(nextState)

        #initializes these values in each counter so that they exist in the counter 
        #When we access them later.
        _forgettable = self.values[(state, action)]
        _forgettable = self.values[(nextState, nextAction)]
        _forgettable = self.eligibility[(state, action)]
        _forgettable = self.eligibility[(nextState, nextAction)]

        #store delta for later use
        # delta = current reward + discount*next states Q-value - current state Q-value
        delta = reward + (self.discount * self.values[(nextState, nextAction)]) - self.values[(state, action)]
        
        #update the eligibilty value using dutch trace.
        #discounts the the previous value to reduce the impact of looping
        #actions during exploration
        self.eligibility[(state, action)] = (1 - self.alpha) * self.eligibility[(state, action)] + 1

        #For each (state,action) pair in values, update the values based on 
        #eligibilty and then update the eligibilty 
        for k, v in self.values.iteritems():
            trace = self.eligibility[k]
            self.values[k] = v + (self.alpha * delta * trace)
            self.eligibility[k] = trace * self.discount * self.y

        #if we are at the end of an episode (terminal state) 
        #clear eligibility trace so that new episodes are not
        #compounded on old episodes.
        if not self.getLegalActions(nextState):
            self.eligibility = util.Counter()

    def observeTransition(self, state,action,nextState,deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        nextAction = self.decideAction(nextState)
        self.update(state,action,nextState,deltaReward)

    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        #print state
        #print self.lastState
        if not self.lastState == None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state

#A simple Pacman Sarsa(lambda) agent for pacman.py
#fixes elpsilon, gamma, alpha, and numtraining to 
#be the same as the PacmanQAgent for easy comparison
class PacmanSarsaAgent(SarsaLambdaAgent):
    "Exactly the same as SarsaLambdaAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanSarsaLambdaAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        SarsaLambdaAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of SarsaLambdaAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = SarsaLambdaAgent.getAction(self,state)
        return action


class ApproximateSarsaAgent(PacmanSarsaAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanSarsaAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        val = 0
        for f in features.sortedKeys():
            val += self.weights[f] * features[f]
        self.values[(state,action)] = val
        return val

    #update values and based on the previous values and current 
    #eligibulity trace.  The current (state,action) is updated
    #in eligibility to keep a record to how often an action has
    #has been taken
    def update(self, state, action, nextState, reward):
        
        nextAction = self.decideAction(nextState)
        
        _forgettable = self.values[(state, action)]
        _forgettable = self.values[(nextState, nextAction)]
        _forgettable = self.eligibility[(state, action)]
        _forgettable = self.eligibility[(nextState, nextAction)]

        #store delta for later use
        # delta = current reward + discount*next states Q-value - current state Q-value
        delta = reward + (self.discount * self.values[(nextState,nextAction)]) - self.values[(state, action)]
        
        #update the eligibilty value using dutch trace.
        #discounts the the previous value to reduce the impact of looping
        #actions during exploration
        self.eligibility[(state, action)] = (1 - self.alpha) * self.eligibility[(state, action)] + 1

        #For each (state,action) pair in values, update the values based on 
        #eligibilty and then update the eligibilty 
        for k, v in self.values.iteritems():
            trace = self.eligibility[k]
            self.eligibility[k] = trace * self.discount * self.y
            
            #update feature vector
            features = self.featExtractor.getFeatures(k[0], k[1])
            for f in features.sortedKeys():
                self.weights[f] = self.weights[f] + (self.alpha * delta * self.eligibility[k])

        #clear eligibility trace so that new episodes are not
        #compounded on old episodes.
        if not self.getLegalActions(nextState):
            self.eligibility = util.Counter()

        return 

