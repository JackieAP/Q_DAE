import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# =========================
# SETUP
# =========================
simulator = AerSimulator()
BASE_URL = "https://api.climatetrace.org"

# =========================
# FETCH DATA
# =========================
def fetch_emissions(country):
    print(f"\n🌍 Fetching emissions for {country}...\n")
    data = {}
    for year in range(2015, 2023):
        try:
            r = requests.get(
                f"{BASE_URL}/v7/sources/emissions",
                params={"year": year, "gas": "co2e_100yr", "gadmId": country},
                timeout=10
            )
            js = r.json()
            ts = js.get("totals", {}).get("timeseries", [])
            total = sum(x.get("emissionsQuantity", 0) for x in ts)
            data[year] = total
            print(f"{year}: {total:,.0f}")
        except:
            print(f"{year}: error")
    return data

# =========================
# NORMALISE
# =========================
def normalise(values):
    arr = np.array(values)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

# =========================
# LABELS
# =========================
def create_labels(values):
    return [1 if values[i] > values[i-1] else 0 for i in range(1, len(values))]

# =========================
# QUANTUM MODEL
# =========================
def create_circuit(n):
    def circuit(x, theta):
        qc = QuantumCircuit(n, 1)

        # Encode
        for i in range(n):
            qc.ry(x[i] * np.pi, i)

        # Trainable layer
        for i in range(n):
            qc.rx(theta[i], i)

        # Entanglement
        for i in range(n-1):
            qc.cz(i, i+1)

        qc.measure(0,0)
        return qc

    return circuit

# =========================
# RUN
# =========================
def run(qc):
    result = simulator.run(qc, shots=500).result()
    counts = result.get_counts()
    return counts.get("1",0)/500

# =========================
# LOSS
# =========================
def loss(theta, X, y, circuit):
    return np.mean([(run(circuit(X[i], theta)) - y[i])**2 for i in range(len(X))])

# =========================
# TRAIN (IMPROVED)
# =========================
def train(X, y, circuit, n, steps=200):
    theta = np.random.rand(n)
    best = theta
    best_loss = loss(theta, X, y, circuit)

    for _ in range(steps):
        candidate = best + np.random.normal(0, 0.1, n)
        l = loss(candidate, X, y, circuit)

        if l < best_loss:
            best_loss = l
            best = candidate

    return best

# =========================
# CLASSICAL BASELINE
# =========================
def classical_baseline(X):
    return [np.random.randint(0,2) for _ in X]

# =========================
# ACCURACY
# =========================
def accuracy(pred, y):
    return sum(int(p==t) for p,t in zip(pred,y))/len(y)

# =========================
# DASHBOARD
# =========================
def plot_dashboard(country, years, values, q_acc, c_acc, probs):

    fig = plt.figure(figsize=(15,8))
    gs = gridspec.GridSpec(2,2)

    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    # Panel 1
    ax1.plot(years, values, marker='o', linewidth=2)
    ax1.set_title(f"CO2 Emissions — {country}")
    ax1.grid(True)

    # Panel 2
    ax2.bar(["Quantum","Classical"], [q_acc, c_acc])
    ax2.set_title("Model Accuracy")

    # Panel 3
    ax3.hist(probs, bins=6)
    ax3.set_title("Quantum Output Distribution")

    plt.suptitle("Quantum Climate Intelligence System", fontsize=14)
    plt.show()

# =========================
# MAIN
# =========================
def main():

    print("="*50)
    print("⚛️ QUANTUM CLIMATE INTELLIGENCE SYSTEM")
    print("="*50)

    country = input("\nEnter country code (USA, CHN, IND): ").upper()

    data = fetch_emissions(country)
    years = sorted(data.keys())
    values = [data[y] for y in years]

    # SPLIT (IMPORTANT FIX)
    train_values = values[:6]
    test_values = values[6:]

    train_norm = normalise(train_values)

    # DATASET
    X = [train_norm[i:i+2] for i in range(len(train_norm)-2)]
    y = create_labels(train_values)[1:]

    circuit = create_circuit(2)

    print("\n⚛️ Training quantum model...")
    theta = train(X, y, circuit, 2)

    # Predictions
    q_preds = [1 if run(circuit(x, theta))>0.5 else 0 for x in X]
    c_preds = classical_baseline(X)

    q_acc = accuracy(q_preds, y)
    c_acc = accuracy(c_preds, y)

    # Test prediction
    test = normalise(values[-2:])
    prob = run(circuit(test, theta))

    # =========================
    # TERMINAL OUTPUT
    # =========================
    print("\n"+"="*50)
    print(f"RESULTS — {country}")
    print("="*50)

    print("\n📉 Normalised Pattern:")
    for y_, n in zip(years, normalise(values)):
        print(f"{y_}: {n:.2f} {'█'*int(n*20)}")

    print("\n⚛️ Parameters:", theta)

    print("\n🔮 Prediction:")
    print(f"Probability of increase: {prob:.2%}")
    print("→ RISING" if prob>0.5 else "→ FALLING")

    confidence = abs(prob-0.5)*2
    print(f"Confidence: {confidence:.2%}")

    print("\n📊 Accuracy:")
    print(f"Quantum:   {q_acc:.2%}")
    print(f"Classical: {c_acc:.2%}")

    # Circuit display
    print("\n⚛️ Sample Quantum Circuit:")
    print(circuit(X[0], theta).draw())

    # Dashboard
    probs = [run(circuit(x, theta)) for x in X]
    plot_dashboard(country, years, values, q_acc, c_acc, probs)

if __name__ == "__main__":
    main()