import numpy as np
X = [1,2,3,4,5,6,7,8,9]
Y = [1,2,3,4,5,6,7,8,9]
X = np.array(X)
Y = np.array(Y)
namenet = "Processed/dataset_10M.npz"
print('End of dataset reached @: ',len(X),'\n','Saved as: ',namenet)
np.savez(namenet, X, Y)