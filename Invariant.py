import numpy as np


def StrainEnergyDensity(I,c1):
    W = c1* ( I[1]/I[2] + 2*np.sqrt(I[2])-5)
    return W


def compute_B(F):
    F_trans = F.transpose()
    B = np.matmul(F,F_trans)
    return B

def invariants_B_and_C(A):
    I_1 = np.trace(A)
    I_2 = 0.5 * ( I_1**2 - np.trace(np.matmul(A,A)))
    I_3 = np.linalg.det(A)
    I = np.array ([I_1,I_2,I_3])
    return I

def compute_P(B,W,F,I):
    F_trans = F.transpose()
    term_one = 2*F*W[0]
    term_two = W[1]*( 2*I[0]*F - 2*np.matmul(B,F))
    term_three = 2*I[2]*W[2]*np.linalg.inv(F_trans)
    P = term_three+term_two+term_one
    return P



def compute_C(F):
    F_transp = F.transpose()
    C = np.matmul(F_transp,F)
    return C

def derivatives_W(c1,I):
    W_I1 = 0
    W_I2 = c1/I[2]
    W_I3 = -c1*I[1]/I[2]**2 + c1/(np.sqrt(I[2]))
    W_I = np.array([W_I1,W_I2,W_I3])
    return W_I

def eta_coeffs(I,W_I):
    eta_0 = 2* ( I[0]*W_I[1] + I[1]*W_I[2])
    eta_1 = -2* ( W_I[1] + I[0]*W_I[2])
    eta_2 = 2*W_I[2]
    eta = np.array([eta_0, eta_1,eta_2])
    return eta

def eta_coeffs_sigma(I,c1):
    eta_0 = -c1*I[1]/I[2] + c1/np.sqrt(I[2])
    eta_1 = 2*I[0]*c1/(I[1]*I[2])
    eta_2 = 2*c1/(I[2]**2)
    eta = np.array([eta_0, eta_1,eta_2])
    return eta

def Piola_2nd_opt1(prediction,C,answer):
    S_nn = prediction[0]*np.identity(3) + prediction[1]*C + prediction[2]*np.matmul(C,C)
    S_a = answer[0][0][0]*np.identity(3) + answer[0][0][1]*C + answer[0][0][2]*np.matmul(C,C)
    return S_nn,S_a


def Piola_2nd_opt2(prediction,answer,F):
    S_nn = np.matmul(np.linalg.inv(F),prediction)
    S_a = np.matmul(np.linalg.inv(F),answer)
    return S_nn, S_a

def sigma(S,F,I_3):
    temp = np.matmul(S,F.transpose())
    sig  = (1/I_3)* np.matmul(F,temp)
    return sig