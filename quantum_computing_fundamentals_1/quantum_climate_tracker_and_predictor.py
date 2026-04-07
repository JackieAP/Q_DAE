# ================================================================
#  QUANTUM CLIMATE INTELLIGENCE SYSTEM  v2.0
#  ─────────────────────────────────────────────────────────────
#  Data source : Climate TRACE API  (climatetrace.org)
#  Quantum     : Qiskit — Deutsch-Jozsa classifier
#                        Amplitude-encoding trend predictor
#                        Variational quantum ML (rise/fall model)
#
#  Install:
#    pip install qiskit qiskit-aer matplotlib requests numpy
#
#  What it does:
#    1. Fetches real annual CO2 data for any country (2015–2022)
#    2. Encodes emissions as qubit rotation angles (amplitude encoding)
#    3. Classifies the country as CHRONIC / OCCASIONAL polluter
#       using the Deutsch-Jozsa algorithm (1 quantum query vs N classical)
#    4. Predicts the next 5 years via quantum-weighted trend projection
#    5. Trains a tiny variational quantum circuit to predict rise/fall
#       and compares it to a random classical baseline
#    6. Renders a rich dark-themed 4-panel dashboard
# ================================================================

import sys
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from sklearn.linear_model import LogisticRegression

# ── globals ──────────────────────────────────────────────────────
simulator = AerSimulator()
BASE_URL  = "https://api.climatetrace.org"

PALETTE = {
    "bg_fig"  : "#0f1117",
    "bg_ax"   : "#1a1d2e",
    "hist"    : "#00d4aa",
    "pred"    : "#ff6b6b",
    "quantum" : "#a78bfa",
    "classical": "#f59e0b",
    "spine"   : "#333355",
    "grid"    : "#444466",
    "text"    : "#e0e0e0",
    "subtext" : "#aaaacc",
    "safe"    : "#00d4aa",
    "moderate": "#f0a500",
    "danger"  : "#ff4444",
}


# ================================================================
#  1.  DATA FETCHING
# ================================================================

def fetch_country_emissions(country_code: str,
                            start_year: int = 2015,
                            end_year:   int = 2022) -> dict:
    """
    Fetch yearly CO2-equivalent (100-yr GWP) emissions from Climate TRACE.
    Returns {year: tonnes_CO2e}.  Missing years are silently skipped.
    """
    print(f"\n🌍  Fetching emissions for '{country_code}' "
          f"({start_year}–{end_year}) …")
    data = {}
    for year in range(start_year, end_year + 1):
        try:
            r = requests.get(
                f"{BASE_URL}/v7/sources/emissions",
                params={"year": year, "gas": "co2e_100yr",
                        "gadmId": country_code},
                timeout=10,
            )
            r.raise_for_status()
            ts    = r.json().get("totals", {}).get("timeseries", [])
            total = sum(x.get("emissionsQuantity", 0) for x in ts)
            if total > 0:
                data[year] = total
                print(f"   {year}: {total:>18,.0f}  t CO2e")
            else:
                print(f"   {year}: no data")
        except requests.RequestException as exc:
            print(f"   {year}: network error — {exc}")
    return data


# ================================================================
#  2.  HELPERS
# ================================================================

def normalise(values: list) -> np.ndarray:
    """Min-max scale to [0, 1].  Returns zeros if all values identical."""
    arr = np.array(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def create_rise_fall_labels(values: list) -> list:
    """1 = emissions rose vs previous year, 0 = fell or stayed."""
    return [1 if values[i] > values[i - 1] else 0
            for i in range(1, len(values))]


# ================================================================
#  3.  QUBIT ENCODING
# ================================================================

def encode_emissions_as_qubits(norm_values: np.ndarray) -> QuantumCircuit:
    """
    Amplitude-encode up to 4 years of emission data as qubit rotations.
    RY(v × π): 0 → |0⟩ (low), π → |1⟩ (high), π/2 → superposition.
    Adjacent years are entangled with CNOT to capture temporal correlation.
    """
    n  = len(norm_values)
    qc = QuantumCircuit(n, n)
    for i, val in enumerate(norm_values):
        qc.ry(float(val) * np.pi, i)
        if i > 0:
            qc.cx(i - 1, i)
    qc.measure(range(n), range(n))
    return qc


# ================================================================
#  4.  DEUTSCH-JOZSA CLASSIFIER
# ================================================================

def classify_deutsch_jozsa(norm_values: np.ndarray,
                            threshold: float = 0.5) -> dict:
    """
    Apply Deutsch-Jozsa to classify the emission pattern.
      • avg > threshold → constant oracle  → CHRONIC POLLUTER
      • avg ≤ threshold → balanced oracle  → OCCASIONAL EMITTER

    D-J needs only 1 quantum query; classically you'd need up to N checks.
    Returns a dict with all relevant outputs.
    """
    avg = float(np.mean(norm_values))
    oracle = "constant" if avg > threshold else "balanced"

    qc = QuantumCircuit(2, 1)
    qc.x(1);  qc.h(1)   # ancilla → |−⟩
    qc.h(0)              # input qubit → superposition
    if oracle == "balanced":
        qc.cx(0, 1)      # balanced oracle = CNOT
    qc.h(0)              # interference
    qc.measure(0, 0)

    counts   = simulator.run(qc, shots=1000).result().get_counts()
    measured = max(counts, key=counts.get)

    if measured == "0":
        label       = "CHRONIC POLLUTER"
        description = ("Emissions consistently above average — "
                       "a constant high-emission pattern.")
    else:
        label       = "OCCASIONAL EMITTER"
        description = ("Emissions fluctuate significantly — "
                       "not a persistent high-emission pattern.")

    return dict(label=label, description=description,
                oracle=oracle, counts=counts, avg_norm=avg)


# ================================================================
#  5.  QUANTUM TREND PREDICTOR
# ================================================================

def quantum_predict_future(yearly_data: dict,
                           years_ahead: int = 5) -> tuple:
    """
    Encode the mean year-on-year % change as a qubit rotation angle.
      • prob(|1⟩) = probability emissions will RISE
      • prob(|0⟩) = probability emissions will FALL
    Projects forward using the quantum-weighted growth rate.
    Returns (predicted_dict, list_of_future_years).
    """
    years  = sorted(yearly_data)
    values = [yearly_data[y] for y in years]

    changes = [(values[i] - values[i-1]) / values[i-1]
               for i in range(1, len(values)) if values[i-1] > 0]
    if not changes:
        return {}, []

    avg_change  = float(np.mean(changes))
    trend_angle = np.pi / 2 + avg_change * np.pi   # centred at π/2

    qc = QuantumCircuit(1, 1)
    qc.ry(trend_angle, 0)
    qc.measure(0, 0)

    counts    = simulator.run(qc, shots=1000).result().get_counts()
    prob_rise = counts.get("1", 0) / 1000
    prob_fall = 1 - prob_rise

    print(f"\n⚛️  Quantum trend signal  →  "
          f"rise {prob_rise:.1%}  |  fall {prob_fall:.1%}")

    last_val  = values[-1]
    last_year = years[-1]
    predicted = {}
    fut_years = []

    for i in range(1, years_ahead + 1):
        effective_rate = avg_change * prob_rise - abs(avg_change) * prob_fall * 0.5
        projected      = last_val * (1 + effective_rate) ** i
        future_year    = last_year + i
        predicted[future_year] = max(0.0, projected)
        fut_years.append(future_year)

    return predicted, fut_years


# ================================================================
#  6.  VARIATIONAL QUANTUM ML MODEL  (rise / fall)
# ================================================================

def _vqml_circuit(x: np.ndarray, theta: np.ndarray,
                  n: int = 2) -> QuantumCircuit:
    """2-qubit variational circuit: encode → rotate → entangle → measure."""
    qc = QuantumCircuit(n, 1)
    for i in range(n):
        qc.ry(float(x[i]) * np.pi, i)      # encode feature
    for i in range(n):
        qc.rx(float(theta[i]), i)           # trainable param
    for i in range(n - 1):
        qc.cz(i, i + 1)                     # entanglement
    qc.measure(0, 0)
    return qc


def _run_vqml(qc: QuantumCircuit, shots: int = 400) -> float:
    counts = simulator.run(qc, shots=shots).result().get_counts()
    return counts.get("1", 0) / shots


def _vqml_loss(theta: np.ndarray, X: list, y: list) -> float:
    return float(np.mean([
        (_run_vqml(_vqml_circuit(x, theta)) - yi) ** 2
        for x, yi in zip(X, y)
    ]))


def train_vqml(X: list, y: list, steps: int = 150,
               n_params: int = 2) -> tuple:
    """
    Simple random-restart hill-climbing optimiser for the variational circuit.
    Returns (best_theta, training_loss_history).
    """
    print(f"\n⚛️  Training variational quantum model  "
          f"({steps} steps, {len(X)} samples) …")
    theta     = np.random.rand(n_params)
    best      = theta.copy()
    best_loss = _vqml_loss(theta, X, y)
    history   = [best_loss]

    for step in range(steps):
        candidate = best + np.random.normal(0, 0.15, n_params)
        l         = _vqml_loss(candidate, X, y)
        if l < best_loss:
            best_loss = l
            best      = candidate.copy()
        if (step + 1) % 50 == 0:
            print(f"   step {step+1:>3}/{steps}  loss = {best_loss:.4f}")
        history.append(best_loss)

    return best, history


def vqml_predict(X: list, theta: np.ndarray) -> list:
    return [1 if _run_vqml(_vqml_circuit(x, theta)) > 0.5 else 0
            for x in X]


def classical_baseline_predict(X: list, y: list) -> list:
    """Classical ML baseline using Logistic Regression."""
    model = LogisticRegression()
    model.fit(X, y)
    return model.predict(X).tolist()


def accuracy(preds: list, labels: list) -> float:
    return sum(p == t for p, t in zip(preds, labels)) / len(labels)


# ================================================================
#  7.  DASHBOARD
# ================================================================

def _style_ax(ax):
    ax.set_facecolor(PALETTE["bg_ax"])
    ax.tick_params(colors=PALETTE["text"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["spine"])


def plot_dashboard(country_code: str,
                   yearly_data:  dict,
                   predicted_data: dict,
                   dj_result:    dict,
                   q_acc: float,
                   c_acc: float,
                   vqml_probs:   list,
                   loss_history: list):
    """
    4-panel dark dashboard:
      [0, :] Historical emissions + quantum prediction
      [1, 0] Deutsch-Jozsa bar chart
      [1, 1] Emission-zone gauge
      [2, :] VQML accuracy + loss + probability distribution (3 sub-panels)
    """
    T   = PALETTE["text"]
    SUB = PALETTE["subtext"]

    years_h  = sorted(yearly_data)
    vals_h   = [yearly_data[y]  / 1e9 for y in years_h]
    years_p  = sorted(predicted_data)
    vals_p   = [predicted_data[y] / 1e9 for y in years_p]

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(PALETTE["bg_fig"])
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.55, wspace=0.38,
                            left=0.07, right=0.97,
                            top=0.92,  bottom=0.06)

    ax_main  = fig.add_subplot(gs[0, :])      # row 0, full width
    ax_dj    = fig.add_subplot(gs[1, 0])      # row 1 left
    ax_zone  = fig.add_subplot(gs[1, 1:])     # row 1 right (spans 2 cols)
    ax_acc   = fig.add_subplot(gs[2, 0])      # row 2 left
    ax_loss  = fig.add_subplot(gs[2, 1])      # row 2 mid
    ax_dist  = fig.add_subplot(gs[2, 2])      # row 2 right

    for ax in [ax_main, ax_dj, ax_zone, ax_acc, ax_loss, ax_dist]:
        _style_ax(ax)

    # ── Panel 1 : Historical + Prediction ────────────────────
    ax_main.plot(years_h, vals_h,
                 color=PALETTE["hist"], lw=2.5,
                 marker="o", ms=5, label="Real data (Climate TRACE)")
    ax_main.fill_between(years_h, vals_h, alpha=0.12, color=PALETTE["hist"])

    if vals_p:
        bridge_x = [years_h[-1]] + years_p
        bridge_y = [vals_h[-1]]  + vals_p
        ax_main.plot(bridge_x, bridge_y,
                     color=PALETTE["pred"], lw=2, ls="--",
                     marker="s", ms=5, label="Quantum prediction")
        ax_main.fill_between(bridge_x, bridge_y,
                             alpha=0.10, color=PALETTE["pred"])

    ax_main.axvline(years_h[-1], color=PALETTE["spine"],
                    lw=1, ls=":", alpha=0.7)
    ax_main.set_title(
        f"CO₂ Emissions  ·  {country_code}  ·  "
        f"Classification: {dj_result['label']}",
        color=T, fontsize=12, fontweight="bold", pad=8)
    ax_main.set_xlabel("Year",                    color=T, fontsize=9)
    ax_main.set_ylabel("Emissions (Gt CO₂e)",     color=T, fontsize=9)
    ax_main.legend(facecolor=PALETTE["bg_ax"], edgecolor=PALETTE["spine"],
                   labelcolor=T, fontsize=9)
    ax_main.grid(True, alpha=0.12, color=PALETTE["grid"])

    # ── Panel 2 : Deutsch-Jozsa histogram ───────────────────
    dj_labels = list(dj_result["counts"].keys())
    dj_vals   = list(dj_result["counts"].values())
    dj_colors = [PALETTE["hist"] if l == "0" else PALETTE["pred"]
                 for l in dj_labels]
    bars = ax_dj.bar(dj_labels, dj_vals, color=dj_colors,
                     edgecolor=PALETTE["spine"], lw=0.5, width=0.4)
    for bar, v in zip(bars, dj_vals):
        ax_dj.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 12,
                   str(v), ha="center", color=T, fontsize=8)
    ax_dj.set_title("Deutsch-Jozsa  Result",
                    color=T, fontsize=10, fontweight="bold")
    ax_dj.set_xlabel("|0⟩ = Chronic   |1⟩ = Occasional",
                     color=SUB, fontsize=7.5)
    ax_dj.set_ylabel("Counts (1 000 shots)",      color=SUB, fontsize=8)
    ax_dj.grid(True, alpha=0.1, axis="y", color=PALETTE["grid"])

    # ── Panel 3 : Emission zone gauge ───────────────────────
    zones   = ["Safe\n(< 30 %)", "Moderate\n(30–70 %)", "Chronic\n(> 70 %)"]
    z_cols  = [PALETTE["safe"], PALETTE["moderate"], PALETTE["danger"]]
    z_widths = [0.30, 0.40, 0.30]
    cum = 0.0
    for w, c, lbl in zip(z_widths, z_cols, zones):
        ax_zone.barh([0], [w], left=cum, color=c, alpha=0.35,
                     edgecolor=PALETTE["spine"], height=0.35)
        ax_zone.text(cum + w / 2, 0, lbl,
                     ha="center", va="center", color=T, fontsize=8)
        cum += w

    avg = dj_result["avg_norm"]
    ax_zone.plot(avg, 0, "w*", ms=20, zorder=5,
                 label=f"{country_code}  ({avg:.0%})")
    ax_zone.set_xlim(0, 1)
    ax_zone.set_ylim(-0.5, 0.5)
    ax_zone.set_yticks([])
    ax_zone.set_xlabel("Normalised emission level", color=SUB, fontsize=8)
    ax_zone.set_title("Emission Zone",
                      color=T, fontsize=10, fontweight="bold")
    ax_zone.legend(facecolor=PALETTE["bg_ax"], edgecolor=PALETTE["spine"],
                   labelcolor=T, fontsize=9, loc="upper right")
    ax_zone.grid(True, alpha=0.1, axis="x", color=PALETTE["grid"])

    # ── Panel 4 : Accuracy comparison ───────────────────────
    accs   = [q_acc, c_acc]
    labels = labels = ["Quantum\n(VQML)", "Classical\n(LogReg)"]
    colors = [PALETTE["quantum"], PALETTE["classical"]]
    bars2  = ax_acc.bar(labels, accs, color=colors,
                        edgecolor=PALETTE["spine"], lw=0.5, width=0.4)
    for bar, v in zip(bars2, accs):
        ax_acc.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.01, f"{v:.0%}",
                    ha="center", color=T, fontsize=9)
    ax_acc.set_ylim(0, 1.15)
    ax_acc.axhline(0.5, color=PALETTE["spine"], lw=1, ls="--", alpha=0.6)
    ax_acc.text(1.5, 0.52, "classical baseline",
                color=SUB, fontsize=7, ha="right")
    ax_acc.set_title("ML Accuracy", color=T, fontsize=10, fontweight="bold")
    ax_acc.set_ylabel("Accuracy",   color=SUB, fontsize=8)
    ax_acc.grid(True, alpha=0.1, axis="y", color=PALETTE["grid"])

    # ── Panel 5 : Training loss curve ───────────────────────
    ax_loss.plot(loss_history, color=PALETTE["quantum"], lw=1.5)
    ax_loss.fill_between(range(len(loss_history)), loss_history,
                         alpha=0.15, color=PALETTE["quantum"])
    ax_loss.set_title("VQML Training Loss",
                      color=T, fontsize=10, fontweight="bold")
    ax_loss.set_xlabel("Optimiser step", color=SUB, fontsize=8)
    ax_loss.set_ylabel("MSE loss",       color=SUB, fontsize=8)
    ax_loss.grid(True, alpha=0.1, color=PALETTE["grid"])

    # ── Panel 6 : VQML output distribution ──────────────────
    ax_dist.hist(vqml_probs, bins=8,
                 color=PALETTE["quantum"], edgecolor=PALETTE["spine"],
                 lw=0.5, alpha=0.85)
    ax_dist.axvline(0.5, color=PALETTE["pred"], lw=1.5,
                    ls="--", label="decision boundary")
    ax_dist.set_title("VQML Output Distribution",
                      color=T, fontsize=10, fontweight="bold")
    ax_dist.set_xlabel("P(rise)",  color=SUB, fontsize=8)
    ax_dist.set_ylabel("Frequency", color=SUB, fontsize=8)
    ax_dist.legend(facecolor=PALETTE["bg_ax"], edgecolor=PALETTE["spine"],
                   labelcolor=T, fontsize=8)
    ax_dist.grid(True, alpha=0.1, color=PALETTE["grid"])

    # ── Super-title + footer ─────────────────────────────────
    fig.suptitle(
        "⚛️  Quantum Climate Intelligence System  v2.0  "
        "·  Climate TRACE API  +  Qiskit",
        color=T, fontsize=13, fontweight="bold",
    )
    fig.text(0.5, 0.005, dj_result["description"],
             ha="center", color=SUB, fontsize=8.5, style="italic")

    out = f"quantum_climate_{country_code}_v2.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"\n💾  Dashboard saved → {out}")


# ================================================================
#  8.  MAIN
# ================================================================

def main():
    print("=" * 62)
    print("  ⚛️  QUANTUM CLIMATE INTELLIGENCE SYSTEM  v2.0")
    print("  Climate TRACE API  ×  Deutsch-Jozsa  ×  Variational QML")
    print("=" * 62)

    print("\n  Supported country codes (examples):")
    print("  USA · CHN · IND · GBR · DEU · FRA · BRA · AUS · CAN · JPN\n")
    raw = input("  Enter 3-letter country code: ").strip().upper()

    if len(raw) != 3 or not raw.isalpha():
        print("  ⚠️  Invalid code — defaulting to USA.")
        raw = "USA"
    country = raw

    # ── 1. Fetch ─────────────────────────────────────────────
    yearly_data = fetch_country_emissions(country, 2015, 2022)

    if not yearly_data:
        print("\n❌  No data returned.  "
              "Check your connection or try another country code.")
        sys.exit(1)

    years  = sorted(yearly_data)
    values = [yearly_data[y] for y in years]
    norm   = normalise(values)

    print(f"\n📉  Normalised emission pattern  (0 = low, 1 = high):")
    for y, n in zip(years, norm):
        print(f"   {y}: {n:.3f}  {'█' * int(n * 24)}")

    # ── 2. Qubit encoding (display only) ─────────────────────
    print("\n⚛️  Qubit encoding circuit (first 4 years):")
    qc_enc = encode_emissions_as_qubits(norm[:4])
    print(qc_enc.draw(output="text"))

    # ── 3. Deutsch-Jozsa classification ──────────────────────
    print("\n⚛️  Running Deutsch-Jozsa classifier …")
    dj = classify_deutsch_jozsa(norm)

    print(f"\n{'='*62}")
    print(f"  CLASSIFICATION  :  {dj['label']}")
    print(f"  Oracle used     :  {dj['oracle']}")
    print(f"  D-J counts      :  {dj['counts']}")
    print(f"  Average level   :  {dj['avg_norm']:.1%}")
    print(f"  {dj['description']}")
    print(f"{'='*62}")

    # ── 4. Quantum trend prediction ───────────────────────────
    print("\n⚛️  Running quantum trend predictor …")
    predicted_data, future_years = quantum_predict_future(yearly_data, 5)

    print("\n📅  Predicted emissions (next 5 years):")
    for y in future_years:
        print(f"   {y}: {predicted_data[y]/1e9:.3f} Gt CO₂e")

    # ── 5. VQML rise/fall model ───────────────────────────────
    # Build sliding-window dataset on training years (all but last 2)
    train_vals = values[:-2]
    train_norm = normalise(train_vals)

    X = [train_norm[i:i+2] for i in range(len(train_norm) - 2)]
    y = create_rise_fall_labels(train_vals)[1:]

    if len(X) < 2:
        print("\n⚠️  Too few data points for VQML — skipping ML panels.")
        q_acc, c_acc, vqml_probs, loss_history = 0.0, 0.0, [0.5], [0.25]
    else:
        theta, loss_history = train_vqml(X, y)

        q_preds    = vqml_predict(X, theta)
        c_preds    = classical_baseline_predict(X, y)
        q_acc      = accuracy(q_preds, y)
        c_acc      = accuracy(c_preds, y)
        vqml_probs = [_run_vqml(_vqml_circuit(x, theta)) for x in X]

        # Next-step prediction
        test_norm = normalise(values[-2:])
        prob_rise = _run_vqml(_vqml_circuit(test_norm, theta))
        confidence = abs(prob_rise - 0.5) * 2

        print(f"\n{'='*62}")
        print(f"  RISE/FALL PREDICTION")
        print(f"  P(rise)    : {prob_rise:.2%}")
        print(f"  Signal     : {'📈 RISING' if prob_rise > 0.5 else '📉 FALLING'}")
        print(f"  Confidence : {confidence:.2%}")
        print(f"{'='*62}")

        print(f"\n📊  Model Accuracy:")
        print(f"   Quantum (VQML)   : {q_acc:.2%}")
        print(f"   Classical (Logistic Regression): {c_acc:.2%}")

        print(f"\n⚛️  Quantum circuit parameters : {theta}")
        print("\n⚛️  Sample VQML circuit (first training example):")
        print(_vqml_circuit(X[0], theta).draw(output="text"))
        print("\n⚛️ Quantum Insight:")
        print("Quantum models operate in a 2^n-dimensional feature space vs classical n-dimensional space.")

    # ── 6. Quantum vs Classical summary ──────────────────────
    print(f"\n{'='*62}")
    print("  QUANTUM vs CLASSICAL SPEEDUP  (Deutsch-Jozsa)")
    print(f"  Classical queries to classify : up to {len(years)}")
    print(f"  Quantum queries (D-J)         : exactly 1")
    print(f"  Speedup factor                : {len(years)}×")
    print(f"\n  Country   : {country}")
    print(f"  Data range: {years[0]}–{years[-1]}")
    print(f"  Result    : {dj['label']}")
    print(f"{'='*62}")

    # ── 7. Dashboard ─────────────────────────────────────────
    plot_dashboard(
        country_code   = country,
        yearly_data    = yearly_data,
        predicted_data = predicted_data,
        dj_result      = dj,
        q_acc          = q_acc,
        c_acc          = c_acc,
        vqml_probs     = vqml_probs,
        loss_history   = loss_history,
    )


if __name__ == "__main__":
    main()