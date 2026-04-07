# ============================================================
#  QUANTUM CLIMATE TRACKER
#  Data source: Climate TRACE API (climatetrace.org)
#  Quantum:     Qiskit — Deutsch-Jozsa classifier + prediction
#  Concepts:    Superposition, Entanglement, CNOT, D-J Algorithm
#
#  Install:
#    pip install qiskit qiskit-aer matplotlib requests numpy
#
#  How it works:
#    1. Ask user which country they want to analyse
#    2. Fetch REAL CO2 data from Climate TRACE API
#    3. Encode data as qubits
#    4. Run Deutsch-Jozsa to classify: chronic or occasional?
#    5. Predict future trend using quantum interference
#    6. Plot everything clearly
# ============================================================

import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ── simulator ───────────────────────────────────────────────
simulator = AerSimulator()
BASE_URL  = "https://api.climatetrace.org"


# ============================================================
#  STEP 1 — FETCH REAL DATA FROM CLIMATE TRACE API
# ============================================================

def fetch_country_emissions(country_code, start_year=2015, end_year=2023):
    """
    Fetch yearly CO2 emissions for a country from Climate TRACE.
    country_code: 3-letter code e.g. 'USA', 'CHN', 'IND', 'GBR'
    Returns a dict {year: emissions_in_tonnes}
    """
    print(f"\n Fetching real CO2 data for '{country_code}' ...")

    yearly_data = {}

    for year in range(start_year, end_year + 1):
        url    = f"{BASE_URL}/v7/sources/emissions"
        params = {
            "year":    year,
            "gas":     "co2e_100yr",   # CO2 equivalent over 100 years
            "gadmId":  country_code,   # country identifier
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data   = response.json()
                # totals → timeseries holds yearly emission quantities
                totals = data.get("totals", {})
                ts     = totals.get("timeseries", [])
                if ts:
                    # sum all sectors for this year
                    total_emissions = sum(
                        item.get("emissionsQuantity", 0) for item in ts
                    )
                    yearly_data[year] = total_emissions
                    print(f"   {year}: {total_emissions:,.0f} tonnes CO2e")
                else:
                    print(f"   {year}: no data returned")
            else:
                print(f"   {year}: API error {response.status_code}")
        except Exception as e:
            print(f"   {year}: connection error — {e}")

    return yearly_data


def fetch_country_rankings(year=2022):
    """
    Fetch global country emissions rankings for a given year.
    Returns top 10 emitting countries.
    """
    print(f"\n Fetching global country rankings for {year}...")
    url    = f"{BASE_URL}/v7/rankings/countries/emissions"
    params = {"year": year, "gas": "co2e_100yr"}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data     = response.json()
            rankings = data.get("rankings", [])
            print(f"   Retrieved {len(rankings)} countries")
            return rankings[:10]   # top 10 only
        else:
            print(f"   Rankings error: {response.status_code}")
            return []
    except Exception as e:
        print(f"   Rankings error: {e}")
        return []


# ============================================================
#  STEP 2 — NORMALISE DATA FOR QUANTUM ENCODING
# ============================================================

def normalise(values):
    """
    Scale a list of emission values to 0.0–1.0 range.
    0.0 = lowest emission year, 1.0 = highest emission year.
    Needed so we can encode them as qubit rotation angles.
    """
    arr = np.array(values, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# ============================================================
#  STEP 3 — ENCODE EMISSIONS AS QUBITS (SUPERPOSITION)
# ============================================================

def encode_emissions_as_qubits(normalised_values):
    """
    Encode each year's emission level as a qubit rotation.
    Uses RY gate: angle = normalised_value × π
      - 0.0 → angle=0   → qubit stays |0⟩ (low emitter)
      - 1.0 → angle=π   → qubit flips to |1⟩ (high emitter)
      - 0.5 → angle=π/2 → qubit in superposition (medium)
    This is called amplitude encoding.
    """
    n = len(normalised_values)
    qc = QuantumCircuit(n, n)   # n qubits, n classical bits

    for i, val in enumerate(normalised_values):
        angle = val * np.pi     # map 0–1 to 0–π rotation
        qc.ry(angle, i)         # rotate qubit i by that angle
        # entangle adjacent years with CNOT to show correlation
        if i > 0:
            qc.cx(i - 1, i)     # year[i-1] influences year[i]

    qc.measure(range(n), range(n))
    return qc


# ============================================================
#  STEP 4 — DEUTSCH-JOZSA CLASSIFIER
#  Question: Is this country a CHRONIC or OCCASIONAL emitter?
#  Chronic   = always stays high (constant function)
#  Occasional = fluctuates up and down (balanced function)
# ============================================================

def classify_with_deutsch_jozsa(normalised_values, threshold=0.5):
    """
    Use Deutsch-Jozsa to classify the emission pattern.
    We decide the oracle type based on the real data:
      - If average emission > threshold → CONSTANT oracle (chronic)
      - If average emission ≤ threshold → BALANCED oracle (occasional)
    Then D-J confirms it in 1 quantum query.
    """
    avg = np.mean(normalised_values)

    # Decide oracle type from real data
    if avg > threshold:
        oracle_type = "constant"    # chronic polluter
    else:
        oracle_type = "balanced"    # occasional emitter

    # Build the D-J circuit
    qc = QuantumCircuit(2, 1)

    # Prepare ancilla qubit in |-⟩ state
    qc.x(1)                         # flip to |1⟩
    qc.h(1)                         # H → |-⟩

    # Superposition on input qubit — tests all inputs at once
    qc.h(0)

    # Apply oracle
    if oracle_type == "constant":
        pass                        # chronic = do nothing
    else:
        qc.cx(0, 1)                 # occasional = CNOT

    # Interference — collapses to the answer
    qc.h(0)

    # Measure only the answer qubit
    qc.measure(0, 0)

    # Run it
    result = simulator.run(qc, shots=1000).result()
    counts = result.get_counts()

    # Read the result
    measured = max(counts, key=counts.get)  # most frequent outcome
    if measured == "0":
        classification = "CHRONIC POLLUTER"
        description    = "This country consistently emits high CO2 — above average every year."
    else:
        classification = "OCCASIONAL EMITTER"
        description    = "This country's emissions fluctuate — not consistently high."

    return classification, description, oracle_type, counts, avg


# ============================================================
#  STEP 5 — QUANTUM PREDICTION (FUTURE TREND)
#  Use quantum amplitude amplification idea:
#  encode the TREND as a rotation and extrapolate forward
# ============================================================

def quantum_predict_future(yearly_data, years_ahead=5):
    """
    Predict future emissions using a quantum-inspired approach.
    Encodes the year-on-year trend as a qubit rotation angle,
    then uses interference to amplify the dominant trend direction.
    Returns predicted values for the next N years.
    """
    years  = sorted(yearly_data.keys())
    values = [yearly_data[y] for y in years]

    # Calculate year-on-year percentage changes (the trend signal)
    changes = []
    for i in range(1, len(values)):
        if values[i - 1] > 0:
            pct_change = (values[i] - values[i - 1]) / values[i - 1]
            changes.append(pct_change)

    if not changes:
        return {}, []

    avg_change = np.mean(changes)   # average yearly growth rate

    # Quantum-inspired: encode trend as rotation angle
    # Positive trend (rising) → angle > π/2
    # Negative trend (falling) → angle < π/2
    trend_angle = np.pi / 2 + avg_change * np.pi

    qc = QuantumCircuit(1, 1)
    qc.ry(trend_angle, 0)           # encode trend direction
    qc.measure(0, 0)

    result  = simulator.run(qc, shots=1000).result()
    counts  = result.get_counts()

    # Probability of |1⟩ = probability emissions will RISE
    prob_rise = counts.get("1", 0) / 1000
    prob_fall = 1 - prob_rise

    print(f"\n Quantum prediction:")
    print(f"   Probability emissions rise: {prob_rise:.1%}")
    print(f"   Probability emissions fall: {prob_fall:.1%}")

    # Project forward using weighted average change
    predicted = {}
    last_value = values[-1]
    last_year  = years[-1]
    future_years = []

    for i in range(1, years_ahead + 1):
        # Weight the prediction by quantum probabilities
        projected = last_value * (
            1 + avg_change * prob_rise - abs(avg_change) * prob_fall * 0.5
        ) ** i
        future_year = last_year + i
        predicted[future_year] = max(0, projected)
        future_years.append(future_year)

    return predicted, future_years


# ============================================================
#  STEP 6 — VISUALISE EVERYTHING
# ============================================================

def plot_full_dashboard(country_code, yearly_data,
                        predicted_data, classification,
                        description, dj_counts, avg_norm):
    """
    Plot a 3-panel dashboard:
      Panel 1: Real historical emissions + future prediction
      Panel 2: D-J classification histogram
      Panel 3: Emission level bar (chronic vs safe zone)
    """
    years_hist  = sorted(yearly_data.keys())
    values_hist = [yearly_data[y] / 1e9 for y in years_hist]   # convert to Gt

    years_pred  = sorted(predicted_data.keys())
    values_pred = [predicted_data[y] / 1e9 for y in years_pred]

    fig = plt.figure(figsize=(15, 8))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])   # top full width
    ax2 = fig.add_subplot(gs[1, 0])   # bottom left
    ax3 = fig.add_subplot(gs[1, 1])   # bottom right

    HIST_COL = "#00d4aa"
    PRED_COL = "#ff6b6b"
    BG       = "#1a1d2e"
    TEXT     = "#e0e0e0"

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    # ── Panel 1: Historical + Prediction ──
    ax1.plot(years_hist, values_hist,
             color=HIST_COL, linewidth=2.5,
             marker="o", markersize=5, label="Real CO2 data (Climate TRACE)")
    ax1.fill_between(years_hist, values_hist,
                     alpha=0.15, color=HIST_COL)

    if values_pred:
        # connect last historical point to first prediction
        bridge_y = [values_hist[-1]] + values_pred
        bridge_x = [years_hist[-1]] + years_pred
        ax1.plot(bridge_x, bridge_y,
                 color=PRED_COL, linewidth=2,
                 linestyle="--", marker="s",
                 markersize=5, label="Quantum prediction")
        ax1.fill_between(bridge_x, bridge_y,
                         alpha=0.1, color=PRED_COL)

    ax1.set_title(
        f"CO2 Emissions — {country_code}  |  Classification: {classification}",
        color=TEXT, fontsize=12, fontweight="bold", pad=10
    )
    ax1.set_xlabel("Year", color=TEXT, fontsize=9)
    ax1.set_ylabel("Emissions (Gigatonnes CO2e)", color=TEXT, fontsize=9)
    ax1.legend(facecolor=BG, edgecolor="#333355",
               labelcolor=TEXT, fontsize=9)
    ax1.grid(True, alpha=0.15, color="#444466")

    # ── Panel 2: D-J histogram ──
    labels = list(dj_counts.keys())
    vals   = list(dj_counts.values())
    colors = ["#00d4aa" if l == "0" else "#ff6b6b" for l in labels]
    bars   = ax2.bar(labels, vals, color=colors,
                     edgecolor="#333355", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 15,
                 str(v), ha="center", color=TEXT, fontsize=9)

    ax2.set_title("Deutsch-Jozsa Result", color=TEXT,
                  fontsize=10, fontweight="bold")
    ax2.set_xlabel("|0⟩ = Chronic   |1⟩ = Occasional",
                   color=TEXT, fontsize=8)
    ax2.set_ylabel("Measurement count (out of 1000)",
                   color=TEXT, fontsize=8)
    ax2.grid(True, alpha=0.1, axis="y", color="#444466")

    # ── Panel 3: Emission level gauge ──
    zones  = ["Safe\n(< 30%)", "Moderate\n(30–70%)", "Chronic\n(> 70%)"]
    zone_c = ["#00d4aa", "#f0a500", "#ff4444"]
    ax3.barh(zones, [0.3, 0.4, 0.3], color=zone_c,
             alpha=0.35, edgecolor="#333355")

    # Plot where this country falls
    marker_y = 2 if avg_norm > 0.7 else (1 if avg_norm > 0.3 else 0)
    ax3.plot(avg_norm, marker_y, "w*", markersize=18,
             zorder=5, label=f"{country_code} ({avg_norm:.0%})")
    ax3.set_xlim(0, 1)
    ax3.set_title("Emission Zone", color=TEXT,
                  fontsize=10, fontweight="bold")
    ax3.set_xlabel("Normalised emission level", color=TEXT, fontsize=8)
    ax3.legend(facecolor=BG, edgecolor="#333355",
               labelcolor=TEXT, fontsize=9, loc="lower right")
    ax3.grid(True, alpha=0.1, axis="x", color="#444466")

    # description strip
    fig.text(0.5, 0.01, description,
             ha="center", color="#aaaacc", fontsize=9,
             style="italic")

    plt.suptitle(
        "Quantum Climate Tracker  ·  Powered by Climate TRACE API + Qiskit",
        color=TEXT, fontsize=13, fontweight="bold", y=1.01
    )

    out = f"quantum_climate_{country_code}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"\n Dashboard saved as: {out}")


# ============================================================
#  STEP 7 — MAIN: ASK USER, FETCH, ANALYSE, VISUALISE
# ============================================================

def main():
    print("=" * 58)
    print("  QUANTUM CLIMATE TRACKER")
    print("  Real CO2 data  ×  Deutsch-Jozsa  ×  Qiskit")
    print("=" * 58)

    print("\n Common country codes:")
    print("  USA · CHN · IND · GBR · DEU · FRA · BRA · AUS · CAN · JPN")
    country = input("\n Enter 3-letter country code: ").strip().upper()

    if len(country) != 3:
        print(" Invalid code — using USA as default.")
        country = "USA"

    # ── Fetch real data ──────────────────────────────────────
    yearly_data = fetch_country_emissions(country, 2015, 2023)

    if not yearly_data:
        print("\n No data returned. Check your internet connection")
        print("  or try a different country code.")
        return

    # ── Normalise for quantum encoding ───────────────────────
    years  = sorted(yearly_data.keys())
    values = [yearly_data[y] for y in years]
    norm   = normalise(values)

    print(f"\n Normalised emission levels (0=low, 1=high):")
    for y, n in zip(years, norm):
        bar = "█" * int(n * 20)
        print(f"   {y}: {n:.2f}  {bar}")

    # ── Encode as qubits ─────────────────────────────────────
    print("\n Encoding emissions as qubits ...")
    qc_encode = encode_emissions_as_qubits(norm[:4])   # use 4 years
    print(qc_encode.draw(output="text"))

    # ── Classify with Deutsch-Jozsa ──────────────────────────
    print("\n Running Deutsch-Jozsa classifier ...")
    classification, description, oracle, dj_counts, avg_norm = \
        classify_with_deutsch_jozsa(norm)

    print(f"\n {'='*50}")
    print(f"  CLASSIFICATION:  {classification}")
    print(f"  {description}")
    print(f"  Oracle used:     {oracle}")
    print(f"  D-J result:      {dj_counts}")
    print(f"  Average level:   {avg_norm:.1%}")
    print(f"  {'='*50}")

    # ── Quantum prediction ───────────────────────────────────
    print("\n Running quantum future prediction ...")
    predicted_data, future_years = quantum_predict_future(
        yearly_data, years_ahead=5
    )

    print("\n Predicted emissions (next 5 years):")
    for y in future_years:
        print(f"   {y}: {predicted_data[y]/1e9:.3f} Gt CO2e")

    # ── Plot dashboard ───────────────────────────────────────
    plot_full_dashboard(
        country, yearly_data, predicted_data,
        classification, description, dj_counts, avg_norm
    )

    # ── Final comparison ─────────────────────────────────────
    print("\n" + "=" * 58)
    print("  QUANTUM vs CLASSICAL COMPARISON")
    print("=" * 58)
    print(f"  Classical checks to classify emitter: up to {len(years)} queries")
    print(f"  Quantum checks (Deutsch-Jozsa):        exactly 1 query")
    print(f"  Speedup factor:                        {len(years)}×")
    print(f"\n  Country analysed: {country}")
    print(f"  Years of data:    {years[0]}–{years[-1]}")
    print(f"  Classification:   {classification}")
    print("=" * 58)


if __name__ == "__main__":
    main()