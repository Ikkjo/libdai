# A CS AI student notices that people who drive SUVs (S) consume large amounts
# of gas (G) and are involved in more accidents (A) than the national average.
# In this problem, uppercase letters denote the variable names, lowercase
# values denote the variables with value. For example, P(a,∼s) is the same
# thing as P(A=t,S=f). She has constructed the following Bayesian network:
#
#                          (S)
#                         /  \
#                        /    \
#                       ↓      ↓
#                      (A)    (G)
#
#     P(S = t) = 0.3, P(S = f) = 0.7
#     P(A = t | S = t) = 0.6, P(A = f | S = t) = 0.4
#     P(A = t | S = f) = 0.3, P(A = f | S = f) = 0.7
#     P(G = t | S = t) = 0.7, P(G = f | S = t) = 0.3
#     P(G = t | S = f) = 0.2, P(G = f | S = f) = 0.8
#
#    (a) Compute P(A):
#        i.   By generating the entire joint distribution over these variables
#             and explicitly summing the appropreate values

import dai

S = dai.Var(0, 2)  # Define binary variable SUV (with label 0)
A = dai.Var(1, 2)  # Define binary variable Accident (with label 1)
G = dai.Var(2, 2)  # Define binary variable Gas (with label 2)

# Define probability distribution for S
P_S = dai.Factor(S)
P_S[0] = 0.7            # S = 0
P_S[1] = 0.3            # S = 1

# Define conditional probability of A given S
P_A_given_S = dai.Factor(dai.VarSet(A, S))
P_A_given_S[0] = 0.7    # A = 0, S = 0
P_A_given_S[2] = 0.3    # A = 1, S = 0
P_A_given_S[1] = 0.4    # A = 0, S = 1
P_A_given_S[3] = 0.6    # A = 1, S = 1

# Define conditional probability of G given S
P_G_given_S = dai.Factor(dai.VarSet(G, S))
P_G_given_S[0] = 0.8    # G = 0, S = 0
P_G_given_S[2] = 0.2    # G = 1, S = 0
P_G_given_S[1] = 0.3    # G = 0, S = 1
P_G_given_S[3] = 0.7    # G = 1, S = 1

# Build factor graph consisting of those four factors
SUVAccidentFactors = dai.VecFactor()
SUVAccidentFactors.append(P_S)
SUVAccidentFactors.append(P_G_given_S)
SUVAccidentFactors.append(P_A_given_S)
SUVAccidentNetwork = dai.FactorGraph(SUVAccidentFactors)

# Write factorgraph to a file
SUVAccidentNetwork.WriteToFile('suv_accident.fg')
print('SUV Accident network written to suv_accident.fg')
# Output some information about the factorgraph
print("Factor graph info:")
print("\t", SUVAccidentNetwork.nrVars(), 'variables')
print("\t", SUVAccidentNetwork.nrFactors(), 'factors')

# Define the PropertySet for BP
opts = dai.PropertySet()
opts["maxiter"] = "10000"  # Set maximum number of iterations for BP
opts["tol"] = "1e-6"  # Set tolerance for convergence
opts["updates"] = "SEQRND"  # Sequential random updates
opts["logdomain"] = "1"  # Use log-space updates (more numerically stable)
opts["damping"] = "0.1"  # Optional: Set damping factor for stability

# Initialize the BP inference algorithm with the FactorGraph and PropertySet
bp = dai.BP(SUVAccidentNetwork, opts)

# Run the Belief Propagation algorithm
bp.init()
bp.run()

# Compute and print the belief for variable A (marginalized)
P_A = bp.belief(A)

# Output the result
print("Inference result with BP algorithm:")
print("\tP(A=f) =", P_A[0])  # P(A=f)
print("\tP(A=t) =", P_A[1])  # P(A=t)


# Calculate joint probability of all variables: P(S, A, G)
P = dai.Factor()
for x in range(SUVAccidentNetwork.nrFactors()):
    P *= SUVAccidentNetwork.factor(x)
# Alternate way of calculation joint probability of P(S, A, G)
# joint = P_S * P_A_given_S * P_G_given_S
# P_A = joint.marginal(dai.VarSet(A))

# Not necessary: a Bayesian network is already normalized by definition
P.normalize()

# # Calculate some probabilities
denom = P.marginal(dai.VarSet(A))
print("Inference result using joint probability:")
print('\tP(A=f) =', denom[0])
print('\tP(A=t) =', denom[1])
print('\tP(A=t | S=t) =', P.marginal(dai.VarSet(S, A))[3] / denom[0])
