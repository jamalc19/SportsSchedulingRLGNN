import pickle
import pandas as pd
import os
import csv
import re
import matplotlib.pyplot as plt

files = [f for f in os.listdir('AnalysisPresentation/')]
results={}
for file in files:
    print(file)
    r = pickle.load(open('AnalysisPresentation/' + file, 'rb'))
    if 'Test' in file:
        instances='Test'
    else:
        instances='Train'
    if 'RLGNN' in file:
        model = 'NewS2V' if 'NewS2V' in file else 'OriginalS2V'
        t = re.search("T=[1-4]", file).group()[-1]
        method = 'ImitationLearning' if 'Imitation' in file else ('RestrictedActionSpace' if 'Restricted' in file else 'BaseRL')
        for m in r:
            iter = re.search('([0-9]){2,}',m.replace('100Imitation','Imitation').replace('Teams_T3','Teams')).group()
            if int(iter)>1000:
                print(m)
            results[(instances,model, method, t, iter)] = pd.DataFrame(r[m]).transpose().mean().values[1:]
    else:
        model = file
        t = None
        method=None
        iter=None
        results[(instances,model, method, t, iter)] = pd.DataFrame(r).transpose().mean().values[1:]




results = pd.DataFrame(results).transpose()
results.columns = ['%feasible','%optimal','averagepartialsolution']
results.index.set_names(['Instances','Model','TrainingMethod','T','Episode'],inplace=True)
results.to_csv('4TeamResults.csv')



df = pd.read_csv('4TeamResults.csv')
baselines = df[~df['Model'].isin(['NewS2V','OriginalS2V'])]

traindf = df[df['Instances']=='Train']
testdf = df[df['Instances']=='Test']
bestmodels=traindf[traindf.groupby(['Model','TrainingMethod','T'])['%feasible'].transform(max)==traindf['%feasible']]
bestmodels = bestmodels.groupby(['Model','TrainingMethod','T']).first()

bestmodels = bestmodels.join(testdf.set_index(['Model','TrainingMethod','T','Episode']), how='left',on=['Model','TrainingMethod','T','Episode'], lsuffix='_train',rsuffix='_test')


baselines.loc[baselines['Model'].str.contains('Greedy'),'Model']='Greedy'
baselines.loc[baselines['Model'].str.contains('Random'),'Model']='Random'
baselines.loc[baselines['Model'].str.contains('Hard'),'Model']='NewS2VSize1HardCode'
baselines = baselines[baselines['Instances']=='Train'].set_index('Model')[['%feasible','%optimal','averagepartialsolution']].join(baselines[baselines['Instances']=='Test'].set_index('Model')[['%feasible','%optimal','averagepartialsolution']], how='left', on=['Model'], lsuffix='_train',rsuffix='_test')

resulttable = pd.concat([bestmodels.reset_index(),baselines.reset_index()]).drop(['Instances_train','Instances_test'],axis=1)
resulttable.to_csv('ResultTable.csv')


for i, subdf in df.groupby(['Model','TrainingMethod','T']):
    train = subdf[subdf['Instances']=='Train']
    test = subdf[subdf['Instances'] == 'Test']
    plt.plot(train['Episode'], (train['%feasible'] * 100).astype(int), label='training set')
    plt.plot(test['Episode'], (test['%feasible'] * 100).astype(int), label='testing set')
    plt.xlabel('Episode')
    plt.ylabel('FF')
    plt.ylim(0,100)
    plt.title("Model: {}, T: {}, Training Method: {}".format(i[0],int(i[2]),i[1]))
    plt.legend()
    plt.savefig("Figures/TrainingCurve{}.png".format(i))
    plt.close()

