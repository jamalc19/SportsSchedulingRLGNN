import pickle, os
from Graph import Graph
import torch
from GreedyAgent import GreedyAgent
from RandomAgent import RandomAgent

from RLAgent import RLAgent


def evaluate(agent,instances,restricted_action_space=False):
    results={}
    for instance in instances:
        graph = pickle.load(open(instance, 'rb'))
        done = False
        cumulative_reward = -graph.costconstant
        while not done:
            node_to_add = agent.greedy_action(graph,restricted_action_space)
            reward, done = graph.selectnode(node_to_add,restricted_action_space)
            cumulative_reward += reward
        feasible = len(graph.solution) == graph.solutionsize
        optimal = cumulative_reward==0
        partial_solution = len(graph.solution)/graph.solutionsize
        results[instance] = (len(graph.teams),feasible,optimal, partial_solution)
    return results

def trainingcurve(agent, traininstances,testinstances, modelparams,restricted_action_space=False):
    testresults={}
    trainresults={}
    for params in modelparams:
        print(params)
        agent.model.load_state_dict(torch.load(params))
        testresults[params] = evaluate(agent,testinstances,restricted_action_space)
        trainresults[params] = evaluate(agent, traininstances,restricted_action_space)
    return testresults, trainresults


if __name__=='__main__':


    #mediumtraininstances = ['TrainingInstancesMedium/'+f for f in os.listdir('TrainingInstancesMedium')]
    #mediumtestinstances = ['TestingInstancesMedium/' + f for f in os.listdir('TestingInstancesMedium')]

    smalltraininstances = ['TrainingInstances4teams/'+f for f in os.listdir('TrainingInstances4teams')]
    smalltestinstances = ['TestInstances4teams/' + f for f in os.listdir('TestInstances4teams')]


    #Medium Params
    #MediumImitationNewS2V = ['ModelParams/'+f for f in os.listdir('ModelParams') if 'ImitationLearning' in f]
    #MediumT4 = ['ModelParams/'+f for f in os.listdir('ModelParams') if 'SyntheticInstancesFirstTrain' in f]
    #MediumT1 = ['Jesseparams/' + f for f in os.listdir('Jesseparams') if '1-hop experiment batch size' in f] #From Jesse
    #MediumT1='Jesseparams/1-hop experiment batch size 32200'
    #MediumT4 = 'ModelParams/SyntheticInstancesFirstTrain100'
    #MediumImitationNewS2V = 'ModelParams/ImitationLearningFirstAttempt170'

    #small params
    # Inconsistent nameing and directories so hard coding location of every experiment
    SmallParamsT4 = ['ModelParams/T=4FourTeamInstances' + str(i) for i in range(20,1000,10)]
    SmallParamsT3 = ['ModelParams_4Teams_3/4Teams_T3' + str(i) for i in range(20, 1000, 10)]
    SmallParamsT2 = ['ModelParams/T=2FourTeamInstances' + str(i) for i in range(20, 1000, 10)]
    SmallParamsT1 = ['ModelParams/T=1FourTeamInstances' + str(i) for i in range(20, 1000, 10)]
    SmallParamsRestrictedT4 = ['ModelParams/T=4FourTeamInstancesRestrictedActionSpace' + str(i) for i in range(20, 1000, 10)]
    SmallParamsRestrictedT3 = ['ModelParams/T=3FourTeamInstancesRestrictedActionSpace' + str(i) for i in range(20, 1000, 10)]
    SmallParamsRestrictedT2 = ['ModelParams/T=2FourTeamInstancesRestrictedActionSpace' + str(i) for i in range(20, 1000, 10)]
    SmallParamsRestrictedT1 = ['ModelParams/T=1FourTeamInstancesRestrictedActionSpace' + str(i) for i in range(20, 1000, 10)]

    SmallParamsImitationT1 = ['ModelParams/S2VT=1FourTeamInstances100ImitationLearning' + str(i) for i in
                               range(20, 1000, 10)]
    SmallParamsImitationT2 = ['ModelParams/S2VT=2FourTeamInstances100ImitationLearning' + str(i) for i in
                               range(20, 1000, 10)]
    SmallParamsImitationT3 = ['ModelParams/S2VT=3FourTeamInstances100ImitationLearning' + str(i) for i in
                               range(20, 1000, 10)]
    SmallParamsImitationT4 = ['ModelParams/S2VT=4FourTeamInstances100ImitationLearning' + str(i) for i in
                               range(20, 1000, 10)]

    SmallParamsT1NewS2V = ['ModelParams/NewS2VT=1FourTeamInstances' + str(i) for i in range(20, 1000, 10)]
    SmallParamsT2NewS2V = ['ModelParams/NewS2VT=2FourTeamInstances' + str(i) for i in range(20, 1000, 10)]
    SmallParamsT3NewS2V = ['ModelParams/NewS2VT=3FourTeamInstances' + str(i) for i in range(20, 1000, 10)]
    SmallParamsT4NewS2V = ['ModelParams/NewS2VT=4FourTeamInstances' + str(i) for i in range(20, 1000, 10)]

    SmallParamsT1NewS2VRestricted = ['ModelParams/NewS2VT=1FourTeamInstancesRestrictedActionSpace' + str(i) for i in range(20, 1000, 10)]
    SmallParamsT2NewS2VRestricted = ['ModelParams/NewS2VT=2FourTeamInstancesRestrictedActionSpace' + str(i) for i in range(20, 1000, 10)]
    SmallParamsT3NewS2VRestricted = ['ModelParams/NewS2VT=3FourTeamInstancesRestrictedActionSpace' + str(i) for i in range(20, 1000, 10)]
    SmallParamsT4NewS2VRestricted = ['ModelParams/NewS2VT=4FourTeamInstancesRestrictedActionSpace' + str(i) for i in range(20, 1000, 10)]

    SmallParamsT1NewS2VImitation = ['ModelParams/NewS2VT=1FourTeamInstances100ImitationLearning' + str(i) for i in range(20, 1000, 10)]
    SmallParamsT2NewS2VImitation = ['ModelParams/NewS2VT=2FourTeamInstances100ImitationLearning' + str(i) for i in range(20, 1000, 10)]
    SmallParamsT3NewS2VImitation = ['ModelParams/NewS2VT=3FourTeamInstances100ImitationLearning' + str(i) for i in range(20, 1000, 10)]
    SmallParamsT4NewS2VImitation = ['ModelParams/NewS2VT=4FourTeamInstances100ImitationLearning' + str(i) for i in range(20, 1000, 10)]

    SmallParamsNewS2VEmbeddingSize1 = RLAgent(S2V='New',EMBEDDING_SIZE=1,S2V_T=1)
    SmallParamsNewS2VEmbeddingSize1.model.load_state_dict(torch.load('ModelParams/Size1NewS2V'))


    smallagents = {'Greedy':(GreedyAgent(),None,False),
                   'Random':(RandomAgent(),None,False),
                   'RLGNNT=1':(RLAgent(S2V_T=1),SmallParamsT1,False),
                   'RLGNNT=2':(RLAgent(S2V_T=2),SmallParamsT2,False),
                   'RLGNNT=3':(RLAgent(S2V_T=3),SmallParamsT3,False),
                   'RLGNNT=4':(RLAgent(S2V_T=4),SmallParamsT4,False),
                   'RLGNNT=1RestrictedActionSpace': (RLAgent(S2V_T=1), SmallParamsRestrictedT1,True),
                   'RLGNNT=2RestrictedActionSpace': (RLAgent(S2V_T=2), SmallParamsRestrictedT2,True),
                   'RLGNNT=3RestrictedActionSpace': (RLAgent(S2V_T=3), SmallParamsRestrictedT3,True),
                   'RLGNNT=4RestrictedActionSpace': (RLAgent(S2V_T=4), SmallParamsRestrictedT4,True),
                   'RLGNNT=1Imitation': (RLAgent(S2V_T=1), SmallParamsImitationT1,False),
                   'RLGNNT=2Imitation': (RLAgent(S2V_T=2), SmallParamsImitationT2,False),
                   'RLGNNT=3Imitation': (RLAgent(S2V_T=3), SmallParamsImitationT3,False),
                   'RLGNNT=4Imitation': (RLAgent(S2V_T=4), SmallParamsImitationT4,False),
                   'RLGNNNewS2VT=1': (RLAgent(S2V='New',S2V_T=1), SmallParamsT1NewS2V, False),
                   'RLGNNNewS2VT=2': (RLAgent(S2V='New',S2V_T=2), SmallParamsT2NewS2V, False),
                   'RLGNNNewS2VT=3': (RLAgent(S2V='New',S2V_T=3), SmallParamsT3NewS2V, False),
                   'RLGNNNewS2VT=4': (RLAgent(S2V='New',S2V_T=4), SmallParamsT4NewS2V, False),
                   'RLGNNNewS2VT=1RestrictedActionSpace': (RLAgent(S2V='New',S2V_T=1), SmallParamsT1NewS2VRestricted, True),
                   'RLGNNNewS2VT=2RestrictedActionSpace': (RLAgent(S2V='New',S2V_T=2), SmallParamsT2NewS2VRestricted, True),
                   'RLGNNNewS2VT=3RestrictedActionSpace': (RLAgent(S2V='New',S2V_T=3), SmallParamsT3NewS2VRestricted, True),
                   'RLGNNNewS2VT=4RestrictedActionSpace': (RLAgent(S2V='New',S2V_T=4), SmallParamsT4NewS2VRestricted, True),
                   'RLGNNNewS2VT=1Imitation': (RLAgent(S2V='New',S2V_T=1), SmallParamsT1NewS2VImitation, False),
                   'RLGNNNewS2VT=2Imitation': (RLAgent(S2V='New',S2V_T=2), SmallParamsT2NewS2VImitation, False),
                   'RLGNNNewS2VT=3Imitation': (RLAgent(S2V='New',S2V_T=3), SmallParamsT3NewS2VImitation, False),
                   'RLGNNNewS2VT=4Imitation': (RLAgent(S2V='New',S2V_T=4), SmallParamsT4NewS2VImitation, False),
                   'NewS2VEmbeddingSize1HardCode': (SmallParamsNewS2VEmbeddingSize1, None, False),
                   }


    SmallParamsNewS2VEmbeddingSize1.model.load_state_dict({
        'theta1.weight': torch.tensor([[1.0]]),
        'theta1complex.weight': torch.tensor([[0.]]),
        'theta2.weight': torch.tensor([[0.]]),
        'thetaQ1.weight': torch.tensor([[-1.0]]),
        'thetaQ2.weight': torch.tensor([[1.0]])
    })
    smallagents = {'NewS2VEmbeddingSize1SecondHardCode': (SmallParamsNewS2VEmbeddingSize1, None, False),
                   }
    #mediumagents = {'Greedy': (GreedyAgent(), None), 'Random': (RandomAgent(), None), 'T=1': (RLAgent(S2V_T=1), MediumT1),'T=4': (RLAgent(S2V_T=4), MediumT4), 'NewS2V':(RLAgent2(),MediumImitationNewS2V)}
    #mediumagents = {'T=1': (RLAgent(S2V_T=1), MediumT1), 'T=4': (RLAgent(S2V_T=4), MediumT4)}


    for method in smallagents:
        print(method)
        agent = smallagents[method][0]
        if smallagents[method][1] is None:#no params
            trainresults = evaluate(agent, smalltraininstances)
            testresults = evaluate(agent, smalltestinstances)
        else:
            testresults, trainresults = trainingcurve(agent, smalltraininstances,smalltestinstances, smallagents[method][1],smallagents[method][2] )
        pickle.dump(testresults,open('AnalysisPresentation/TestSmall'+method+'.pkl','wb'))
        pickle.dump(trainresults, open('AnalysisPresentation/TrainSmall' + method + '.pkl', 'wb'))
    '''
    for method in mediumagents:
        print(method)
        agent = mediumagents[method][0]
        if not mediumagents[method][1] is None:#no params
            agent.model.load_state_dict(torch.load(mediumagents[method][1]))
        trainresults = evaluate(agent, mediumtraininstances)
        testresults = evaluate(agent, mediumtestinstances)
        pickle.dump(testresults,open('Analysis/TestMedium'+method+'.pkl','wb'))
        pickle.dump(trainresults, open('Analysis/TrainMedium' + method + '.pkl', 'wb'))
    '''



