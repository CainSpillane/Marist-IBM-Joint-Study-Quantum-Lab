from qiskit_ibm_runtime import QiskitRuntimeService
import sys
import VQECredMethod3

sys.path.insert(0, "..")

from qiskit_ibm_runtime import UserMessenger


sys.path.insert(0, "..")  # Add source_program directory to the path

QiskitRuntimeService.save_account(
    channel="ibm_quantum", token="60a6f674d4c3af8baad5ad33a982e531af05323b4664c4bf90ef0d6d346bac77b0317ebd2e261f7e4d4e903d5983d3435a32c3e634d0085b89d77cd42209bcfa",
    instance="ibm-q-research-2/marist-ibm-1/main", name="research", overwrite="True")

service = QiskitRuntimeService(
    channel="ibm_quantum", instance="ibm-q-research-2/marist-ibm-1/main")

# Select a backend.
backend = service.backend(
    "ibmq_qasm_simulator", instance="ibm-q-research-2/marist-ibm-1/main")

inputs = {"iterations": 3}

user_messenger = UserMessenger()

VQECredMethod3.main(backend, user_messenger)
