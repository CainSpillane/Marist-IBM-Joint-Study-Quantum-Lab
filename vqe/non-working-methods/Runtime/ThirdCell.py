# Runtime Imports
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE  # new import!!!
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.primitives import Estimator
from qiskit.algorithms.optimizers import SPSA
from qiskit_optimization.applications import Maxcut, Tsp  # DEPRECATED
from qiskit.circuit.library import TwoLocal
from qiskit.tools.visualization import plot_histogram
from qiskit import Aer
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session, Options
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


service = QiskitRuntimeService()
backend = "ibmq_qasm_simulator"


# Old Imports

# New Imports
# Session

""" with Session(backend="ibm_guadalupe") as session:
    Estimator = Estimator(session=session)
    job = Estimator.run()
    print(f"Sampler job ID: {job.job_id()}")
    print(f"Sampler job result:" {job.result()})
    # Close the session only if all jobs are finished and
    # you don't need to run more in the session.
    session.close() """

n = 12  # Number of nodes in graph
G = nx.Graph()
G.add_nodes_from(np.arange(0, n, 1))
elist = [

    (0, 1, 1.19),

    (0, 2, 1.0),

    (0, 3, 1.0),

    (0, 4, 1.19),

    (0, 5, 1.19),

    (0, 6, 1.19),

    (0, 7, 1.0),

    (0, 8, 1.19),

    (0, 9, 1.19),

    (0, 10, 1.19),

    (0, 11, 1.0),

    (1, 0, 1.19),

    (1, 4, 1.19),

    (1, 5, 1.19),

    (1, 6, 1.19),

    (1, 8, 1.19),

    (1, 9, 1.19),

    (1, 10, 1.19),

    (2, 0, 1.0),

    (2, 7, 1.0),

    (3, 0, 1.0),

    (3, 11, 1.0),

    (4, 0, 1.19),

    (4, 1, 1.19),

    (5, 0, 1.19),

    (5, 1, 1.19),

    (6, 0, 1.19),

    (6, 1, 1.19),

    (7, 0, 1.0),

    (7, 2, 1.0),

    (8, 0, 1.19),

    (8, 1, 1.19),

    (9, 0, 1.19),

    (9, 1, 1.19),

    (10, 0, 1.19),

    (10, 1, 1.19),

    (11, 0, 1.0),

    (11, 3, 1.0)]

# tuple is (i,j,weight) where (i,j) is the edge
G.add_weighted_edges_from(elist)

colors = ['r' for node in G.nodes()]
pos = nx.spring_layout(G)


def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600,
                     alpha=.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)


draw_graph(G, colors, pos)

# Computing the weight matrix from the random graph
w = np.zeros([n, n])
for i in range(n):
    for j in range(n):
        temp = G.get_edge_data(i, j, default=0)
        if temp != 0:
            w[i, j] = temp['weight']

# Define our Qiskit Maxcut Instance
max_cut = Maxcut(w)
qp = max_cut.to_quadratic_program()

# Translate to Ising Hamiltonian
qubitOp, offset = qp.to_ising()

# Setup our simulator
np.random.seed(123)
seed = 10598
backend = Aer.get_backend('aer_simulator_statevector')  # replace
aer_estimator = AerEstimator(run_options={"shots": 2048, "seed": 10598})


# Construct VQE
opt = SPSA(maxiter=300)
ansatz = TwoLocal(qubitOp.num_qubits, 'ry', 'cz',
                  reps=5, entanglement='linear')

estimator = Estimator(options={"shots": 2048})
vqe = SamplingVQE(estimator, opt, ansatz)  # Shot based

# vqe = VQE(aer_estimator, ansatz, opt) #using aer


# Run VQE
result = vqe.compute_minimum_eigenvalue(qubitOp)

# print results
x = max_cut.sample_most_likely(result.eigenstate)
print('energy:', result.eigenvalue.real)
print('time:', result.optimizer_time)
print('max-cut objective:', result.eigenvalue.real + offset)
print('solution:', x)
print('solution objective:', qp.objective.evaluate(x))

# plot results
colors = ['r' if x[i] == 0 else 'c' for i in range(n)]
draw_graph(G, colors, pos)
