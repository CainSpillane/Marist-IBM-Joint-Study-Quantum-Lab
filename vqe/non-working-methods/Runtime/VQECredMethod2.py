# General imports
import time
import numpy as np

# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp

# The IBM Qiskit Runtime
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Estimator, Session

# SciPy minimizer routine
from scipy.optimize import minimize

# Plotting functions
import matplotlib.pyplot as plt


from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit import QuantumCircuit
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.primitives import Estimator
from qiskit import QuantumCircuit, transpile

from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal



QiskitRuntimeService.save_account(
    channel="ibm_quantum", token="60a6f674d4c3af8baad5ad33a982e531af05323b4664c4bf90ef0d6d346bac77b0317ebd2e261f7e4d4e903d5983d3435a32c3e634d0085b89d77cd42209bcfa",
    instance="ibm-q-research-2/marist-ibm-1/main", name="research", overwrite="True")

service = QiskitRuntimeService(
    channel="ibm_quantum", instance="ibm-q-research-2/marist-ibm-1/main")

# Select a backend.
backend = service.backend(
    "ibmq_qasm_simulator", instance="ibm-q-research-2/marist-ibm-1/main")

# Building Callback function


def build_callback(ansatz, hamiltonian, estimator, callback_dict):
    """Return callback function that uses Estimator instance,
    and stores intermediate values into a dictionary.

    Parameters:
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance
        callback_dict (dict): Mutable dict for storing values

    Returns:
        Callable: Callback function object
    """

    def callback(current_vector):
        """Callback function storing previous solution vector,
        computing the intermediate cost value, and displaying number
        of completed iterations and average time per iteration.

        Values are stored in pre-defined 'callback_dict' dictionary.

        Parameters:
            current_vector (ndarray): Current vector of parameters
                                      returned by optimizer
        """
        # Keep track of the number of iterations
        callback_dict["iters"] += 1
        # Set the prev_vector to the latest one
        callback_dict["prev_vector"] = current_vector
        # Compute the value of the cost function at the current vector
        callback_dict["cost_history"].append(
            estimator.run(ansatz, hamiltonian, parameter_values=current_vector)
            .result()
            .values[0]
        )
        # Grab the current time
        current_time = time.perf_counter()
        # Find the total time of the execute (after the 1st iteration)
        if callback_dict["iters"] > 1:
            callback_dict["_total_time"] += current_time - \
                callback_dict["_prev_time"]
        # Set the previous time to the current time
        callback_dict["_prev_time"] = current_time
        # Compute the average time per iteration and round it
        time_str = (
            round(callback_dict["_total_time"] /
                  (callback_dict["iters"] - 1), 2)
            if callback_dict["_total_time"]
            else "-"
        )
        # Print to screen on single line
        print(
            "Iters. done: {} [Avg. time per iter: {}]".format(
                callback_dict["iters"], time_str
            ),
            end="\r",
            flush=True,
        )

    return callback


hamiltonian = SparsePauliOp.from_list(
    [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
)

ansatz = EfficientSU2(hamiltonian.num_qubits)
ansatz.draw("mpl")

num_params = ansatz.num_parameters
num_params


def cost_func(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    energy = (
        estimator.run(ansatz, hamiltonian,
                      parameter_values=params).result().values[0]
    )
    return energy


x0 = 2 * np.pi * np.random.random(num_params)

callback_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
    "_total_time": 0,
    "_prev_time": None,
}

with Session(backend=backend):
    estimator = Estimator(options={"shots": int(1e4)})
    callback = build_callback(ansatz, hamiltonian, estimator, callback_dict)
    res = minimize(
        cost_func,
        x0,
        args=(ansatz, hamiltonian, estimator),
        method="cobyla",
        callback=callback,
    )
    
    
    
# Is prev_vector equal to the solution vector?
if all(callback_dict["prev_vector"] == res.x):
    print("Prev_vector is equal to Solution Vector")
else:
    print("Prev_vector is NOT equal to Solution Vector")


# Is iters equal to the total number of function evaluations?
if callback_dict["iters"] == res.nfev:
    print("Iters is equal to # of function evaluations")
else:
    print("Iters is NOT equal to # of function evaluations")

print(res)



""" Other methods 
# define ansatz and optimizer
num_qubits = 2
ansatz = TwoLocal(num_qubits, "ry", "cz")
optimizer = SLSQP(maxiter=100)

# define initial point
init_pt = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]

# hamiltonian/operator --> use SparsePauliOp or Operator

hamiltonian = SparsePauliOp.from_list(
    [
        ("II", -1.052373245772859),
        ("IZ", 0.39793742484318045),
        ("ZI", -0.39793742484318045),
        ("ZZ", -0.01128010425623538),
        ("XX", 0.18093119978423156),
    ]
)


def result_callback(job_id, result):
    print(result)

with Session(service, backend="ibmq_qasm_simulator") as session:

    estimator = Estimator()  # no need to pass the session explicitly
    vqe = VQE(estimator, ansatz, optimizer,
             initial_point=init_pt)
    result = vqe.compute_minimum_eigenvalue(hamiltonian) """



