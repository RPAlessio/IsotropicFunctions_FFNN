import numpy as np
import Invariant


def create_target(Strain_Measure,ism, option, n, c1=500):

    # FOR OPTION ONE, WE ENTER F AND OBTAIN THE ETA COEFS AS TARGET

    if option == 1:
        target = np.zeros((n,1,3))
        if ism == 0:
            for i in range(n):
                F = Strain_Measure[i][0:][0:]
                C = Invariant.compute_C(F)
                I = Invariant.invariants_B_and_C(C)
                W = Invariant.derivatives_W(c1, I)
                Coefs = Invariant.eta_coeffs(I, W)
                target[i][0][0:] = Coefs
            return target
        elif ism == 1:
            for i in range(n):
                C = Strain_Measure[i][0:][0:]
                I = Invariant.invariants_B_and_C(C)
                W = Invariant.derivatives_W(c1, I)
                Coefs = Invariant.eta_coeffs(I, W)
                target[i][0][0:] = Coefs
            return target

    # FOR OPTION TWO, WE ENTER F AND OBTAIN P

    if option == 2:
        target = np.zeros((n, 1, 9))
        if ism == 0:
            for i in range(n):
                B = Invariant.compute_B(Strain_Measure[i][0:][0:])
                I = Invariant.invariants_B_and_C(B)
                W = Invariant.derivatives_W(c1, I)
                P = Invariant.compute_P(B, W, Strain_Measure[i][0:][0:], I)
                P_flat = np.reshape(P, (1, 9))
                target[i][0][0:] = P_flat
            return target
        if ism == 1:
            print('Option to compute 1st Piola from Hencky space was not yet implemented')

            #return target

    # FOR OPTION THREE, WE ENTER F AND OBTAIN W

    if option == 3:
        target = np.zeros((n, 1, 1))

        if ism == 0:
            for i in range(n):
                F = Strain_Measure[i][0:][0:]
                B = Invariant.compute_B(F)
                I = Invariant.invariants_B_and_C(B)
                W = Invariant.StrainEnergyDensity(I, c1)
                target[i][0][0] = W
            return target
        elif ism == 1:
            for i in range(n):
                C = Strain_Measure[i][0:][0:]
                I = Invariant.invariants_B_and_C(C)
                W = Invariant.StrainEnergyDensity(I, c1)
                target[i][0][0] = W
            return target

    if option == 4:
        target = np.zeros((n,1,3))
        if ism == 0:
            for i in range(n):
                F = Strain_Measure[i][0:][0:]
                B = Invariant.compute_B(F)
                I = Invariant.invariants_B_and_C(B)
                Coefs = Invariant.eta_coeffs_sigma(I, c1)
                target[i][0][0:] = Coefs
            return target
        if ism == 1:
            for i in range(n):
                C = Strain_Measure[i][0:][0:]
                I = Invariant.invariants_B_and_C(C)
                Coefs = Invariant.eta_coeffs_sigma(I, c1)
                target[i][0][0:] = Coefs
            return target

def compute_Invariants(F_array,n):
    Invs = np.zeros((n,1,3))
    for i in range(n):
        F = F_array[i][0:][0:]
        C = Invariant.compute_C(F)
        I = Invariant.invariants_B_and_C(C)
        Invs[i][0][0:] = I
    return Invs