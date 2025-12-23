import matplotlib.pyplot as plt
import numpy as np

def graph_evol (df, indicator) :
    for pays in df['Pays'].unique() :
        data =df.loc[df['Pays'] == pays]
        plt.plot(data['Année'],data[indicator],label=pays)

    plt.xlabel('Année')    
    plt.ylabel(indicator)
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(data['Année'].min(), data['Année'].max() + 1, 2))
    plt.tight_layout()
    plt.show()