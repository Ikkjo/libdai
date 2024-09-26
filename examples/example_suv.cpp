#include <dai/alldai.h>
#include <iostream>

using namespace dai;

int main() {
    // Define variables S, A, G with binary states (0: false, 1: true)
    Var S(0, 2); // S has two states
    Var A(1, 2); // A has two states
    Var G(2, 2); // G has two states

    // Factor P(S)
    Factor P_S(S);
    P_S[S(0)] = 0.7; // P(S=f)
    P_S[S(1)] = 0.3; // P(S=t)

    // Factor P(A | S)
    Factor P_A_given_S(S & A);
    P_A_given_S[S(0) & A(0)] = 0.7; // P(A=f | S=f)
    P_A_given_S[S(0) & A(1)] = 0.3; // P(A=t | S=f)
    P_A_given_S[S(1) & A(0)] = 0.4; // P(A=f | S=t)
    P_A_given_S[S(1) & A(1)] = 0.6; // P(A=t | S=t)

    // Factor P(G | S)
    Factor P_G_given_S(S & G);
    P_G_given_S[S(0) & G(0)] = 0.8; // P(G=f | S=f)
    P_G_given_S[S(0) & G(1)] = 0.2; // P(G=t | S=f)
    P_G_given_S[S(1) & G(0)] = 0.3; // P(G=f | S=t)
    P_G_given_S[S(1) & G(1)] = 0.7; // P(G=t | S=t)

    // Construct the factor graph
    FactorGraph fg;
    fg.addFactor(P_S);
    fg.addFactor(P_A_given_S);
    fg.addFactor(P_G_given_S);

    // Compute the joint distribution P(S, A, G)
    Factor joint = P_S * P_A_given_S * P_G_given_S;

    // Marginalize out S and G to compute P(A)
    Factor P_A = joint.marginal(A);

    // Output the result
    std::cout << "P(A=f): " << P_A[A(0)] << std::endl;
    std::cout << "P(A=t): " << P_A[A(1)] << std::endl;

    return 0;
}
  
