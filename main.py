import numpy
import numpy as np
import Invariant
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import CreateTargets
import PostProcess
import CreateInput
import random


option = 1

Input_Strain_Measure = 1

if Input_Strain_Measure == 0:
    n_input = 1000
    n_training = 900
    St_train, input_training, St_test, input_test, J = CreateInput.DefGrad_Sample(n_input, n_training)
else:
    n_training = 900
    n_dir = 100
    n_mag = 10
    n_input = n_dir*n_mag
    j_star = 1.1
    St_train, input_training, St_test, input_test, J = CreateInput.Hencky_Sample(n_training, n_dir, n_mag, j_star)


target_training = np.asarray(CreateTargets.create_target(St_train,Input_Strain_Measure,option,n_training ))

#Input_inv = np.asarray(CreateTargets.compute_Invariants(F_arrays,n_entries_training))


input_tensor = torch.FloatTensor(np.asarray(input_training))
#input_tensor = torch.FloatTensor(Input_inv)
target_tensor = torch.FloatTensor(target_training)

#print(target_tensor)

#CREATING TRAININ DATASET
train_ds = TensorDataset(input_tensor, target_tensor)

batch_size = n_training

train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# ----------------------------CREATING TEST SET------------------------------------------------------------------------

number_tests = 1
random_test =np.array(random.sample(range(n_input-n_training), number_tests))
Test_St = St_test[random_test][0:][0:]
target_test = CreateTargets.create_target(Test_St,Input_Strain_Measure, option,number_tests)
input_test_tensor = torch.FloatTensor(np.asarray(input_test[random_test[0]]))
answer_tensor = torch.FloatTensor(np.asarray(target_test))



#----------------------------------------------------------------------------------------------------------------------


input_size = 9
hidden_size = 16
hidden_size_2 = 16

if option == 1:
    num_classes = 3 # target number -> outputs
elif option ==2:
    num_classes = 9
elif option == 3:
    num_classes = 1
num_epochs = 100000
learning_rate = 0.001

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidde_size,num_classes,hidde_size2):
        super(NeuralNet,self).__init__()
        #creating layers

        #layer one -> non-linear neutrons
        self.l1 = nn.Linear(input_size,hidde_size)
        self.relu = nn.Softplus()

        #layer two -> linear regression for output
        self.l2 = nn.Linear(hidde_size,hidde_size2)
        self.relu2 = nn.Softplus()

        self.l3 = nn.Linear(hidde_size2, num_classes)

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        return out


#creating the instance of class

model = NeuralNet(input_size,hidden_size,num_classes,hidden_size_2)


#loss and optmizer

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# training loop

for epoch in range(num_epochs):

    for xb,yb in train_dl:
        #forward pass

        outputs = model.forward(xb)

        loss = criterion(outputs,yb)

        #backwards pass

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch {epoch+1} / {num_epochs}, loss = {loss.item():.4f}, normalized loss = {loss.item()/yb.mean():.4f}')

prediction =[]

# CHECKING TESTS

with torch.no_grad():

    for i in range(number_tests):
        prediction=model.forward(input_test_tensor[i])

PostProcess.postprocess(option, prediction, answer_tensor, Test_St,Input_Strain_Measure)


