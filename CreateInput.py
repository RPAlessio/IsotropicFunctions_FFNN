import numpy as np
from scipy.linalg import expm
import random

def Hencky_Sample(n_training, n_dir, n_mag,j_star):
    Y_basis = np.zeros((6,3,3))
    E = np.zeros((n_dir*n_mag,3,3))
    J = []
    C = np.zeros((n_dir*n_mag,3,3))
    t = np.linspace(0.1,1,num=n_mag)
    input_training = []
    input_test = []
    n_test = n_dir*n_mag - n_training
    test_samples_id = random.sample(range(n_dir*n_mag), n_test)
    C_input_training = np.zeros((n_training,3,3))
    C_input_test = np.zeros((n_test,3,3))
    dummy = 0
    Y_basis[0][0:][0:] = np.sqrt(1/6)*np.array([[2, 0, 0], [0, -1, 0], [0, 0, -1]])
    Y_basis[1][0:][0:] = np.sqrt(1 / 2) * np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
    Y_basis[2][0:][0:] = np.sqrt(1 / 2) * np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    Y_basis[3][0:][0:] = np.sqrt(1 / 2) * np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    Y_basis[4][0:][0:] = np.sqrt(1 / 2) * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    Y_basis[5][0:][0:] = (np.log(j_star)/3) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    for i in range(n_dir):
        v = np.random.randn(1,6)
        norm = np.linalg.norm(v)
        normalized_v = v/norm
        test = np.linalg.norm(normalized_v)
        for j in range(n_mag):
            mag_vec = t[j]*normalized_v
            test2 = np.linalg.norm(mag_vec)
            for k in range(6):
                E[dummy][0:][0:] = E[dummy][0:][0:] + mag_vec[0][k]*Y_basis[k][0:][0:]
            U = expm(E[dummy][0:][0:])
            C[dummy][0:][0:] = np.matmul(U,U)
            J.append(np.sqrt(np.linalg.det(C[dummy][0:][0:])))
            dummy = dummy + 1

    dummy_samp_test = 0
    dummy_samp_train = 0

    for i in range(n_dir*n_mag):
        if i in (test_samples_id):
            C_input_test[dummy_samp_test][0:][0:] = C[i][0:][0:]
            C_flat_test = np.reshape(C_input_test[dummy_samp_test][0:][0:], (1, 9))
            input_test.append(C_flat_test)
            dummy_samp_test = dummy_samp_test+ 1
        else:
            C_input_training[dummy_samp_train][0:][0:] = C[i][0:][0:]
            C_flat_train = np.reshape(C_input_training[dummy_samp_train][0:][0:], (1, 9))
            input_training.append(C_flat_train)
            dummy_samp_train = dummy_samp_train+ 1



    return C_input_training,input_training,C_input_test,input_test,J

def DefGrad_Sample(n_inputs, n_input_training):
    J = []
    F_arrays = np.zeros((n_inputs, 3, 3))
    t = np.linspace(0.0, 1, n_inputs)
    n_inputs_test = n_inputs - n_input_training
    input_training = []
    input_test = []
    test_samples_id = random.sample(range(n_inputs),n_inputs_test)
    F_input_training = np.zeros((n_input_training, 3, 3))
    F_input_test = np.zeros((n_inputs_test, 3, 3))
    for i in range(n_inputs):
        t_i = t[i]
        F_arrays[i][0][0] = np.exp(t_i)
        F_arrays[i][1][1] = np.exp(t_i)
        F_arrays[i][1][2] = t_i
        F_arrays[i][2][2] = np.exp(-t_i)
        J.append(np.sqrt(np.linalg.det(F_arrays[i][0:][0:])))
    dummy_samp_test = 0
    dummy_samp_train = 0

    for i in range(n_inputs):
        if i in (test_samples_id):
            F_input_test[dummy_samp_test][0:][0:] = F_arrays[i][0:][0:]
            F_flat_test = np.reshape(F_input_test[dummy_samp_test][0:][0:], (1, 9))
            input_test.append(F_flat_test)
            dummy_samp_test = dummy_samp_test+ 1
        else:
            F_input_training[dummy_samp_train][0:][0:] = F_arrays[i][0:][0:]
            F_flat_train = np.reshape(F_input_training[dummy_samp_train][0:][0:], (1, 9))
            input_training.append(F_flat_train)
            dummy_samp_train = dummy_samp_train+ 1

    return F_input_training,input_training,F_input_test,input_test,J