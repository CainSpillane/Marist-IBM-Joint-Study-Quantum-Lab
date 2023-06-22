# General imports
import time
import numpy as np

# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp

# The IBM Qiskit Runtime
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Estimator, Session

from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.primitives import Estimator
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

with Session(service, backend="ibmq_qasm_simulator") as session:

    estimator = Estimator()  # no need to pass the session explicitly
    vqe = VQE(estimator, ansatz, optimizer, initial_point=init_pt)
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    print(result.eigenvalue)
    # job = estimator.run()
    # result = job.result()
   #  print(f"job id: {job.job_id()}")
    print(result)

    session.close()
    


