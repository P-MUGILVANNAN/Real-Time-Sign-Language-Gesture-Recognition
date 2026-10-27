"""
SignVision AI — Scientific Research Experiments Dashboard.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from src.experiment_runner import ExperimentRunner

st.set_page_config(page_title="Research Experiments — SignVision AI", page_icon="🔬", layout="wide")

st.title("🔬 Scientific Research Experiments")
st.markdown("""
Investigation of generalization across unseen signers, background invariance,
temporal speed variation, sequence length, and comparative model architectures.
""")

exp_dir = Path("artifacts/experiments")
plots_dir = Path("artifacts/plots")

top_col1, top_col2 = st.columns([3, 1])
with top_col2:
    if st.button("🚀 Re-Run All Experiments", type="primary", use_container_width=True):
        with st.status("Executing Research Experiment Suite...", expanded=True) as status:
            st.write("Running Experiment A (Cross-User Generalization)...")
            st.write("Running Experiment D (LSTM vs GRU vs Random Forest)...")
            st.write("Running Experiment E (Sequence Length Ablation)...")
            st.write("Running Experiment F (Confidence Threshold Calibration)...")
            try:
                runner = ExperimentRunner()
                runner.run_all_experiments()
                status.update(label="All Experiments Completed!", state="complete", expanded=False)
                st.success("Experiments completed and artifacts updated.")
                st.rerun()
            except Exception as e:
                status.update(label=f"Experiments Failed: {e}", state="error")
                st.error(f"Error: {e}")

tab_a, tab_d, tab_e, tab_f = st.tabs([
    "👥 Exp A: Cross-User Generalization",
    "🤖 Exp D: Architecture Comparison",
    "⏱️ Exp E: Sequence Length Ablation",
    "🎯 Exp F: Confidence Threshold Trade-Off"
])

# ----------------- Tab A: Cross-User Generalization -----------------
with tab_a:
    st.subheader("Experiment A: Signer-Independent vs Signer-Dependent Generalization")
    st.markdown("""
    **Hypothesis**: A model trained with random frame splitting across signers experiences data leakage,
    leading to artificially inflated accuracy that fails on unseen signers. Signer-independent splitting
    demonstrates the true real-world generalization gap.
    """)

    csv_a = exp_dir / "experiment_a_cross_user.csv"
    plot_a = plots_dir / "experiment_a_cross_user.png"

    c1, c2 = st.columns([1, 1])
    with c1:
        if csv_a.is_file():
            df_a = pd.read_csv(csv_a)
            st.dataframe(df_a, use_container_width=True)

            gap = 0.0
            if len(df_a) >= 2:
                si_f1 = df_a[df_a["Condition"].str.contains("Independent")]["Macro_F1"].values[0]
                sd_f1 = df_a[df_a["Condition"].str.contains("Dependent")]["Macro_F1"].values[0]
                gap = sd_f1 - si_f1

            st.metric("Signer Generalization Gap (Δ F1)", f"{gap:+.3f}", delta_color="inverse")
            st.caption("Negative gap represents drop in performance when encountering completely new signers.")
        else:
            st.info("Experiment A data pending. Click 'Re-Run All Experiments'.")

    with c2:
        if plot_a.is_file():
            st.image(str(plot_a), use_container_width=True)

# ----------------- Tab D: Model Architecture Comparison -----------------
with tab_d:
    st.subheader("Experiment D: Model Architecture Comparison (LSTM vs GRU vs Random Forest)")
    st.markdown("""
    Compares the proposed 2-layer LSTM against a 2-layer GRU and a Random Forest baseline
    trained on aggregated temporal statistical landmark features (mean, std, min, max, velocity).
    """)

    csv_d = exp_dir / "experiment_d_model_comparison.csv"
    plot_d = plots_dir / "experiment_d_model_comparison.png"

    c1, c2 = st.columns([1, 1])
    with c1:
        if csv_d.is_file():
            df_d = pd.read_csv(csv_d)
            st.dataframe(df_d, use_container_width=True)
            st.markdown("""
            **Key Insights**:
            - **LSTM** provides optimal temporal context modeling for dynamic multi-phase signs.
            - **GRU** yields slightly lower parameter footprint with competitive CPU latency.
            - **Random Forest** provides an interpretable non-recurrent benchmark.
            """)
        else:
            st.info("Experiment D data pending.")

    with c2:
        if plot_d.is_file():
            st.image(str(plot_d), use_container_width=True)

# ----------------- Tab E: Sequence Length Ablation -----------------
with tab_e:
    st.subheader("Experiment E: Sequence Length Ablation (15 vs 30 vs 45 Frames)")
    st.markdown("""
    Investigates the temporal window length required for reliable sign disambiguation
    versus inference latency.
    """)

    csv_e = exp_dir / "experiment_e_sequence_length.csv"
    plot_e = plots_dir / "experiment_e_sequence_length.png"

    c1, c2 = st.columns([1, 1])
    with c1:
        if csv_e.is_file():
            df_e = pd.read_csv(csv_e)
            st.dataframe(df_e, use_container_width=True)
        else:
            st.info("Experiment E data pending.")

    with c2:
        if plot_e.is_file():
            st.image(str(plot_e), use_container_width=True)

# ----------------- Tab F: Confidence Threshold Trade-Off -----------------
with tab_f:
    st.subheader("Experiment F: Confidence Thresholding (Coverage vs Accuracy)")
    st.markdown("""
    Calibrates the abstention threshold. Higher thresholds increase prediction reliability
    on accepted signs while abstaining ('No confident gesture detected') on uncertain gestures.
    """)

    csv_f = exp_dir / "experiment_f_confidence_threshold.csv"
    plot_f = plots_dir / "experiment_f_confidence_threshold.png"

    c1, c2 = st.columns([1, 1])
    with c1:
        if csv_f.is_file():
            df_f = pd.read_csv(csv_f)
            st.dataframe(df_f, use_container_width=True)
        else:
            st.info("Experiment F data pending.")

    with c2:
        if plot_f.is_file():
            st.image(str(plot_f), use_container_width=True)
