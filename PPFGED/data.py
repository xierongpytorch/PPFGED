import numpy as np
import pandas as pd
df =pd.read_csv("")
print('df.shape:',df.shape)

x = df.drop(columns='label')
y = df['label']

corpus = np.array(x)
corpus
a=corpus[0][0]
d=corpus[0].tolist()


def k_random_response(value, values,epsilon):

    if not isinstance(values, list):
        raise Exception("The values should be list")
    if value not in values:
        raise Exception("Errors in k-random response")
    p = np.e ** epsilon / (np.e ** epsilon + len(values) - 1)
    if np.random.random() <= p:
        return value
    values.remove(value)
    return values[np.random.randint(low=0, high=len(values))]
k_random_response(a, d,0.1)
i=0
j=0
q=[]
p=[]

while i<len(corpus):
    while j<len(corpus[0]):
        c=corpus[i][j]
        b=corpus[i].tolist()
        e=k_random_response(c, b,10)
        q.append(e)
        j=j+1
    p.append(q)
    i=i+1
    j=0
    q=[]
test4=pd.DataFrame(p)
data1 = pd.concat([y,test4], axis=1)
print('data1.shape:',data1.shape)
data1.to_csv('')