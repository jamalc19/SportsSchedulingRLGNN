import pandas as pd

filep = r'C:\Users\bzhan\Desktop\SportsSchedulingRLGNN\Results\Run5_Only4Teams_TestSet4950'
roll1 = pd.read_pickle(filep)

roll1_df = pd.DataFrame.from_dict(roll1)
roll1_df_rewards = roll1_df.apply(lambda x:  [y[0] for y in x])
roll1_df_length = roll1_df.apply(lambda x:  [y[1] for y in x])

solution_length = roll1_df_length.apply(pd.value_counts).fillna(0)
solution_length.loc[(solution_length!=0).any(axis=1)].to_csv('solution_length.csv')

pd.DataFrame((roll1_df_rewards == 0).sum(axis=0)).to_csv('optimal_solutions.csv')

