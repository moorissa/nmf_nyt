# # Topic Modeling for The New York Times News Dataset
# ## Nonnegative Matrix Factorization
# ### Author: Moorissa Tjokro

# ### a. Plot the divergence objective for learning 25 topics as a function of iteration.

# In[447]:

'''read data'''
# contains index of words appearing in that document and the number of times they appear
with open('data/nyt_data.txt') as f:
    documents = f.readlines()
documents = [x.strip().strip('\n').strip("'") for x in documents] 

# contains vocabs with rows as index
with open('data/nyt_vocab.dat') as f:
    vocabs = f.readlines()
vocabs = [x.strip().strip('\n').strip("'") for x in vocabs] 

'''create matrix X'''
numDoc = 8447
numWord = 3012 
X = np.zeros([numWord,numDoc])

for col in range(len(documents)):
    for row in documents[col].split(','):
        X[int(row.split(':')[0])-1,col] = int(row.split(':')[1])


# In[468]:

'''randomly initialize W and H with nonnegative values'''
rank = 25
T = 100
W = np.zeros([numWord,rank])
H = np.zeros([rank,numDoc])

for row in range(numWord):
    W[row] = np.random.rand(rank)
for row in range(rank):
    H[row] = np.random.rand(numDoc)
    
'''setting divergence penalty''' #iterate values in H, then in W
d_iter = np.zeros(100)

for iteration in range(100):
    
    '''iterate all values in H'''
    m1 = np.dot(W.T,X)
    m2 = np.dot(W,H)
    m3 = np.dot(m2.T,W)
    second = np.divide(m1,m3.T + 0.0000000000000001)

    for k in range(rank):
        for j in range(numDoc):
            H[k,j] = np.multiply(H[k,j], second[k,j])
    
    '''iterate all values in W'''
    n1 = np.dot(H,X.T)
    n2 = np.dot(W,H)
    n3 = np.dot(n2,H.T)
    third = np.divide(n1.T,n3 + 0.0000000000000001)

    for i in range(numWord):
        for k in range(rank):
            W[i,k] = np.multiply(W[i,k], third[i,k])
        
    '''plot objective function'''
#     D = np.multiply(X, np.log(1/(n2 + 0.0000000000000001))) + n2
#     d_iter[iteration] = np.sum(D)
    D = np.multiply(X,np.log(n2+0.0000000000000001)) - n2
    d_iter[iteration] = -np.sum(D)

fig= plt.figure(figsize = (15,6))
ax = fig.add_subplot(1,1,1)
ax.plot(range(100),d_iter[:100])
plt.title('Plot of divergence objective in 100 iterations')
plt.ylabel('$D(X||WH)$')
plt.xlabel('iteration $t$')
plt.show()


# ### b. Ten words with the largest weight.

# In[502]:

'''normalize each column to sum to zero'''
W_normed = W / np.sum(W,axis=0)


# In[511]:

'''for each column of W, list the 10 words having the largest weight and show the weight'''
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)    
vList = []

for topic in range(rank):
    v = pd.DataFrame(vocabs)
    v[1] = W_normed[:,topic].round(6)
    v = v.sort([1, 0], ascending=[0,1]).rename(index=int, columns={0: "Topic {}".format(topic+1), 1: "Weight"}).head(10)
    v = v.reset_index(drop=True)
    vList.append(v)
    
for num in [5,10,15,20,25]:
    print('\n',(pd.concat(vList[num-5:num], axis=1)),'\n')
