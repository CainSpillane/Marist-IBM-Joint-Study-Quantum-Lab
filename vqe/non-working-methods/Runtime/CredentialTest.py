from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService


QiskitRuntimeService.save_account(
    channel="ibm_quantum", token="60a6f674d4c3af8baad5ad33a982e531af05323b4664c4bf90ef0d6d346bac77b0317ebd2e261f7e4d4e903d5983d3435a32c3e634d0085b89d77cd42209bcfa",
    instance="ibm-q-research-2/marist-ibm-1/main", name="research", overwrite="True")

""" IBMProvider.save_account(
    token="60a6f674d4c3af8baad5ad33a982e531af05323b4664c4bf90ef0d6d346bac77b0317ebd2e261f7e4d4e903d5983d3435a32c3e634d0085b89d77cd42209bcfa", overwrite="True")""" 


service = QiskitRuntimeService(
    channel="ibm_quantum", instance="ibm-q-research-2/marist-ibm-1/main")


# Create a circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Select a backend.
backend = service.backend(
    "ibmq_qasm_simulator", instance="ibm-q-research-2/marist-ibm-1/main")


# Transpile the circuit
transpiled = transpile(qc, backend=backend)

# Submit a job.
# Get results.
# print(qc.result().get_counts())
