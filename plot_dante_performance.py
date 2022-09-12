
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('/Users/michael/Documents/dante improvement.csv')




plt.bar(df.iloc[:, -5:],np.tile(np.linspace(0,11,12), (5, 1)).T)
plt.show()

barwidth = 0.15
br1 = np.arange(4)
br2 = [x + barwidth for x in br1]
br3 = [x + barwidth for x in br2]
br4 = [x + barwidth for x in br3]
br5 = [x + barwidth for x in br4]






fig = plt.figure(constrained_layout=True, figsize=(7.08661, 3), dpi=100)
figs = fig.subfigures(1, 2, width_ratios=[0.9, 0.1])
axs = figs[0].subplots(3,1, sharey=True)
axs2 = figs[1].subplots(1,1, sharey=True)
axs2.set_xticks([])
axs2.set_yticks([])
axs2.axis('off')



for idx, metric in enumerate(['skill', 'mae', 'corr']):
    axs[idx].bar(br1, 100*df.loc[df['metric'] == metric].iloc[:, 2], linewidth=0, color ='r', width = barwidth, label='2015')
    axs[idx].bar(br2, 100*df.loc[df['metric'] == metric].iloc[:, 3], linewidth=0, color ='b', width = barwidth, label='2016')
    axs[idx].bar(br3, 100*df.loc[df['metric'] == metric].iloc[:, 4], linewidth=0, color ='g', width = barwidth, label='2017')
    axs[idx].bar(br4, 100*df.loc[df['metric'] == metric].iloc[:, 5], linewidth=0, color ='k', width = barwidth, label='2018')
    axs[idx].bar(br5, 100*df.loc[df['metric'] == metric].iloc[:, 6], linewidth=0, color ='k', width = barwidth, label='Average')

    axs[idx].plot([-1,5], [0,0], color='black', linewidth=0.4)
    axs[idx].set_xlim([-0.25, 3.85])
    axs[idx].set_ylabel(metric)
    if idx == 2:
        axs[idx].set_xticks([r + barwidth*1.5 for r in range(4)],
                   ['$\gamma = -7$',
                    '$\gamma = 0$',
                    '$\gamma = 7$',
                    '$\gamma = 14$'])

    else:
        axs[idx].set_xticks([],
                   [])
axs2.legend(*axs[0].get_legend_handles_labels(), loc='center', ncol=1,frameon=False)

plt.show()
