import numpy as np
from numpy.random import Generator, default_rng
from skqulacs.circuit import LearningCircuit
from skqulacs.qnn import QNNClassifier
from sklearn.metrics import f1_score

# TODO: 乱数
random_seed = 0
np.random.seed(random_seed)

def generate_data(bits):
    """Generate training and testing data."""
    n_rounds = 20  # Produces n_rounds * bits datapoints.
    excitations = []
    labels = []
    for n in range(n_rounds):
        for bit in range(bits):
            rng = np.random.uniform(-np.pi, np.pi)
            excitations.append(rng)
            # TODO: スケール設定が必要？
            #labels.append(1 if (-np.pi / 2) <= rng <= (np.pi / 2) else -1)
            labels.append(1 if (-np.pi / 2) <= rng <= (np.pi / 2) else 0)

    split_ind = int(len(excitations) * 0.7)
    train_excitations = excitations[:split_ind]
    test_excitations = excitations[split_ind:]

    train_labels = labels[:split_ind]
    test_labels = labels[split_ind:]

    return train_excitations, np.array(train_labels), \
        test_excitations, np.array(test_labels)

def _innser_conv_circuit1(circuit, src, dest):
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RY_gate(src, angle)
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RY_gate(dest, angle)
    circuit.add_CNOT_gate(src, dest)
    return circuit

def _innser_conv_circuitU5(circuit, src, dest):
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RX_gate(src, angle)
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RX_gate(dest, angle)
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RZ_gate(src, angle)
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RZ_gate(dest, angle)

    # TODO: CRZをCNOTに分解
    # qml.CRZ(params[4], wires=[wires[1], wires[0]])
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RZ_gate(dest, angle) #.add_control_qubit(dest, 1)
    circuit.add_CNOT_gate(dest, src)
    # TODO: CRZをCNOTに分解
    # qml.CRZ(params[5], wires=[wires[0], wires[1]])
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RZ_gate(dest, angle) #.add_control_qubit(src, 1)
    circuit.add_CNOT_gate(src, dest)

    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RX_gate(src, angle)
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RX_gate(dest, angle)
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RZ_gate(src, angle)
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RZ_gate(dest, angle)
    angle = np.random.uniform(-np.pi, np.pi)
    return circuit

def _innser_conv_circuitU9(circuit, src, dest):
    circuit.add_H_gate(src)
    circuit.add_H_gate(dest)

    # TODO: CZをCNOTに分解
    #circuit.add_CZ_gate(src, dest)
    circuit.add_CNOT_gate(src, dest)
    circuit.add_Z_gate(dest)

    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RX_gate(src, angle)
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RX_gate(dest, angle)
    return circuit

def _innser_conv_circuitU13(circuit, src, dest):
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RY_gate(src, angle)
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RY_gate(dest, angle)

    # TODO: CRZをCNOTに分解
    # qml.CRZ(params[2], wires=[wires[1], wires[0]])
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RZ_gate(dest, angle)
    circuit.add_CNOT_gate(src, dest)

    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RY_gate(src, angle)
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RY_gate(dest, angle)

    # TODO: CRZをCNOTに分解
    # qml.CRZ(params[5], wires=[wires[0], wires[1]])
    angle = np.random.uniform(-np.pi, np.pi)
    circuit.add_parametric_RZ_gate(dest, angle)
    circuit.add_CNOT_gate(src, dest)

    return circuit

def conv_circuit(circuit, src, dest):
    return _innser_conv_circuit1(circuit, src, dest)
    #return _innser_conv_circuitU5(circuit, src, dest)
    #return _innser_conv_circuitU9(circuit, src, dest)
    #return _innser_conv_circuitU13(circuit, src, dest)

def pooling_circuit(circuit, src, dest):
    angle = np.random.rand()
    circuit.add_parametric_RZ_gate(src, angle)
    circuit.add_CNOT_gate(src, dest)
    angle = np.random.rand()
    circuit.add_parametric_RX_gate(dest, angle)
    circuit.add_X_gate(src)
    circuit.add_CNOT_gate(src, dest)
    return circuit

def create_qcnn_ansatz(
    n_qubit: int, seed: int = 0
) -> LearningCircuit:
    # def preprocess_x(x: List[float], index: int) -> float:
    #     xa = x[index % len(x)]
    #     return xa

    rng = default_rng(seed)
    circuit = LearningCircuit(n_qubit)
    for i in range(n_qubit):
        circuit.add_input_RX_gate(i, lambda x : x)

    # cluster state
    for i in range(n_qubit):
        circuit.add_H_gate(i)
    for this_bit in range(n_qubit):
        next_bit = this_bit + 1 if this_bit < n_qubit - 1 else 0
        #print(f"this_bit: {this_bit} next_bit: {next_bit}")
        circuit.add_CNOT_gate(this_bit, next_bit)
        circuit.add_Z_gate(next_bit)

    # depth 1 (0, 1)
    circuit = conv_circuit(circuit, 0, 1)
    circuit = pooling_circuit(circuit, 0, 1)
    circuit = conv_circuit(circuit, 2, 3)
    circuit = pooling_circuit(circuit, 2, 3)
    circuit = conv_circuit(circuit, 4, 5)
    circuit = pooling_circuit(circuit, 4, 5)
    circuit = conv_circuit(circuit, 6, 7)
    circuit = pooling_circuit(circuit, 6, 7)

    # depth 2 (1, 3)
    circuit = conv_circuit(circuit, 1, 3)
    circuit = pooling_circuit(circuit, 1, 3)
    circuit = conv_circuit(circuit, 5, 7)
    circuit = pooling_circuit(circuit, 5, 7)

    # depth 3 (3, 7)
    circuit = conv_circuit(circuit, 3, 7)
    circuit = pooling_circuit(circuit, 3, 7)

    # # depth 1
    # circuit = conv_circuit(circuit, 0, 4)
    # circuit = pooling_circuit(circuit, 0, 4)
    # circuit = conv_circuit(circuit, 2, 5)
    # circuit = pooling_circuit(circuit, 2, 5)
    # circuit = conv_circuit(circuit, 4, 6)
    # circuit = pooling_circuit(circuit, 4, 6)
    # circuit = conv_circuit(circuit, 6, 7)
    # circuit = pooling_circuit(circuit, 6, 7)

    # # depth 2
    # circuit = conv_circuit(circuit, 4, 6)
    # circuit = pooling_circuit(circuit, 4, 6)
    # circuit = conv_circuit(circuit, 5, 7)
    # circuit = pooling_circuit(circuit, 5, 7)

    # # depth 3
    # circuit = conv_circuit(circuit, 6, 7)
    # circuit = pooling_circuit(circuit, 6, 7)
    return circuit


# qubitの数
nqubit = 8

x_train, y_train, x_test, y_test = generate_data(nqubit)

circuit = create_qcnn_ansatz(nqubit)

num_class = 2
solver="BFGS"
qcl = QNNClassifier(circuit, num_class, solver)

maxiter = 20
opt_loss, opt_params = qcl.fit(x_train, y_train, maxiter)
print("trained parameters: ", opt_params)
print("loss: ", opt_loss)

y_pred = qcl.predict(x_test)
# print("y_pred: ", y_pred)
# print("y_test: ", y_test)
print("f1_score: ", f1_score(y_test, y_pred, average="weighted"))

# maxiter = 4
# epochs=5
# #epochs=10
# for i in range(epochs):
#     print(f"epochs: {i}/{epochs}")
#     opt_loss, opt_params = qcl.fit(x_train, y_train, maxiter)
#     #print("trained parameters: ", opt_params)
#     print("loss: ", opt_loss)
#     y_pred = qcl.predict(x_test)
#     #print("y_pred: ", y_pred)
#     print("f1_score: ", f1_score(y_test, y_pred, average="weighted"))
