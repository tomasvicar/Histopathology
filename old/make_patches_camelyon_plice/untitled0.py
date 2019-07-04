import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

r='C:/Users/tomas/Desktop/population.csv'

df = pd.read_csv(r)

pop=df.Value.tolist()


numbers=np.zeros(9)
for k in pop:
    kk=int(str(k)[0])
    numbers[kk-1]+=1
    
    
    
plt.plot(numbers)