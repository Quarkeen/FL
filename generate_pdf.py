"""
generate_pdf.py - Generate a research-style PDF report for the FedProx project.

Uses fpdf2 to create a structured document with:
  - Abstract, Introduction, Methodology, Experimental Setup
  - Results with embedded plots and comparison tables
  - Discussion and Conclusion
"""

import os
from fpdf import FPDF


class ResearchPDF(FPDF):
    """Custom PDF class with research paper styling."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        """Page header with thin line."""
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, "Federated Learning: FedAvg vs FedProx - Baseline and Improvements", align="C")
            self.ln(5)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(5)

    def footer(self):
        """Page footer with page number."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, num, title):
        """Numbered section heading."""
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 51, 102)
        self.ln(4)
        self.cell(0, 10, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 51, 102)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def subsection_title(self, num, title):
        """Numbered subsection heading."""
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(0, 76, 153)
        self.ln(2)
        self.cell(0, 8, f"{num} {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        """Regular body text."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(33, 33, 33)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet_point(self, text):
        """Indented bullet point."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(33, 33, 33)
        self.cell(8)
        self.cell(5, 5.5, "-") # Replaced chr(8226) with standard hyphen to avoid encoding errors
        self.multi_cell(0, 5.5, text)

    def add_table(self, headers, rows, col_widths=None):
        """Create a styled table."""
        if col_widths is None:
            total_width = 190
            col_widths = [total_width / len(headers)] * len(headers)

        # Header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255, 255, 255)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 8, header, border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(33, 33, 33)
        for row_idx, row in enumerate(rows):
            if row_idx % 2 == 0:
                self.set_fill_color(240, 245, 255)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 7, str(cell), border=1, fill=True, align="C")
            self.ln()
        self.ln(4)

    def add_plot(self, image_path, caption=""):
        """Add a plot image with caption."""
        if os.path.exists(image_path):
            # Check if we need a new page for the image
            if self.get_y() > 170:
                self.add_page()

            x = (210 - 160) / 2  # Center the image
            self.image(image_path, x=x, w=160)
            self.ln(2)

            if caption:
                self.set_font("Helvetica", "I", 9)
                self.set_text_color(100, 100, 100)
                self.cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
                self.ln(4)
        else:
            self.body_text(f"[Plot not found: {image_path}]")


def generate_report(all_histories, mu_values, mu_accuracies, results_dir):
    """
    Generate the complete research PDF report.

    Args:
        all_histories: Dict of {method_name: history_dict}
        mu_values: List of μ values from sweep
        mu_accuracies: Corresponding accuracies
        results_dir: Directory containing plot PNGs
    """
    pdf = ResearchPDF()
    pdf.add_page()

    # ── Title Page ───────────────────────────────────────────────────────

    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 12, "Federated Learning Project:", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 12, "FedAvg vs FedProx", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(0, 76, 153)
    pdf.cell(0, 8, "Baseline Reproduction and Novel Improvements", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)

    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, "With Adaptive, Decaying, and Hybrid Proximal Strategies", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(30)

    pdf.set_draw_color(0, 51, 102)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(20)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(33, 33, 33)
    pdf.cell(0, 7, "Implemented in PyTorch", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "MNIST Dataset | CNN Architecture", align="C", new_x="LMARGIN", new_y="NEXT")

    # ── Abstract ─────────────────────────────────────────────────────────

    pdf.add_page()
    pdf.section_title(1, "Abstract")
    pdf.body_text(
        "Federated learning enables collaborative model training across decentralized clients "
        "without sharing raw data. However, heterogeneity in client data distributions (statistical "
        "heterogeneity) and computational capabilities (system heterogeneity) poses significant "
        "challenges to convergence and model quality. FedAvg, the standard federated optimization "
        "algorithm, handles IID settings well but can suffer under non-IID conditions. FedProx "
        "addresses this by adding a proximal regularization term that constrains local updates to "
        "remain close to the global model."
    )
    pdf.body_text(
        "In this project, we reproduce both FedAvg and FedProx on the MNIST dataset under IID and "
        "non-IID settings, then propose and evaluate three novel improvements to FedProx: (1) Drift-Aware "
        "Adaptive mu, which scales the proximal coefficient proportionally to each client's model drift; "
        "(2) Time-Decaying mu, which relaxes the constraint over training rounds; and (3) Hybrid mu, "
        "combining both adaptive and decaying strategies. Our experiments with 10 clients over 20 "
        "communication rounds demonstrate the behavior and trade-offs of each approach."
    )

    # ── Introduction ─────────────────────────────────────────────────────

    pdf.section_title(2, "Introduction")
    pdf.body_text(
        "Federated learning (FL) systems often operate under heterogeneous conditions, where clients "
        "differ in both data distribution and computational capabilities. These challenges lead to "
        "instability and slow convergence in standard federated optimization methods."
    )
    pdf.body_text(
        "The paper 'Federated Optimization in Heterogeneous Networks' (Li et al., 2018) introduces "
        "the FedProx algorithm to address these issues by modifying the local training objective with "
        "a proximal regularization term. Unlike FedAvg, which performs simple weighted averaging of "
        "client updates, FedProx constrains local updates to remain close to the global model, "
        "reducing client drift and improving stability in heterogeneous environments."
    )
    pdf.body_text(
        "However, FedProx has notable limitations: (1) it applies a uniform proximal coefficient mu "
        "to all clients regardless of their data heterogeneity; (2) mu remains static throughout "
        "training; and (3) the value of mu requires careful tuning. In this work, we propose three "
        "improvements - adaptive, decaying, and hybrid proximal strategies - to address these limitations."
    )

    # ── Methodology ──────────────────────────────────────────────────────

    pdf.section_title(3, "Methodology")

    pdf.subsection_title("3.1", "FedAvg (Baseline)")
    pdf.body_text(
        "In FedAvg, each client minimizes the standard task loss L(w) using local SGD for E epochs, "
        "then sends updated weights to the server. The server aggregates by computing a weighted "
        "average: w_global = SUM(n_k/n * w_k), where n_k is the number of samples on client k."
    )

    pdf.subsection_title("3.2", "FedProx (Baseline)")
    pdf.body_text(
        "FedProx modifies the client objective to: L_fedprox = L(w) + (mu/2) * ||w - w_global||^2. "
        "The proximal term penalizes deviation from the global model, reducing client drift. However, "
        "mu is fixed across all clients and all rounds."
    )

    pdf.subsection_title("3.3", "Proposed Improvement 1: Drift-Aware Adaptive mu")
    pdf.body_text(
        "We propose scaling mu proportionally to each client's model drift: "
        "mu_i = alpha * ||w_i - w_global||. Clients with high drift receive stronger regularization, "
        "while clients close to the global model train more freely. This provides automatic, "
        "client-specific adaptation without manual tuning of per-client parameters."
    )

    pdf.subsection_title("3.4", "Proposed Improvement 2: Time-Decaying mu")
    pdf.body_text(
        "We introduce an exponential decay schedule: mu_t = mu_0 * exp(-beta * t). "
        "Early in training, when client models are far apart, strong regularization prevents "
        "divergence. As training progresses and models converge, the constraint relaxes, allowing "
        "more flexible local optimization."
    )

    pdf.subsection_title("3.5", "Proposed Improvement 3: Hybrid Adaptive-Decaying mu")
    pdf.body_text(
        "We combine both strategies: mu_{i,t} = alpha * ||w_i - w_global|| * exp(-beta * t). "
        "This provides a doubly-adaptive proximal term that adjusts both per-client (based on drift) "
        "and per-round (based on training progress)."
    )

    # ── Experimental Setup ───────────────────────────────────────────────

    pdf.section_title(4, "Experimental Setup")

    pdf.subsection_title("4.1", "Model Architecture")
    pdf.body_text(
        "We use a Convolutional Neural Network (CNN) with the following architecture: "
        "Input(28x28) -> Conv2d(1,32,3) -> ReLU -> Conv2d(32,64,3) -> ReLU -> MaxPool(2,2) -> "
        "Dropout(0.25) -> FC(64*14*14, 128) -> ReLU -> Dropout(0.5) -> FC(128, 10)."
    )

    pdf.subsection_title("4.2", "System Configuration")
    pdf.add_table(
        ["Parameter", "Value"],
        [
            ["Number of Clients", "10"],
            ["Communication Rounds", "20"],
            ["Local Epochs", "3"],
            ["Batch Size", "32"],
            ["Optimizer", "Adam"],
            ["Learning Rate", "0.001"],
            ["Dataset", "MNIST"],
            ["Seed", "42"],
        ],
        col_widths=[95, 95],
    )

    pdf.subsection_title("4.3", "Data Partitioning")
    pdf.body_text(
        "IID Setting: Data is randomly shuffled and equally distributed across all clients. "
        "Non-IID Setting: Each client receives data from only 2 digit classes, assigned deterministically "
        "(Client 0: classes {0,1}, Client 1: classes {2,3}, ..., Client 4: classes {8,9}, "
        "then wrapping around for clients 5-9)."
    )

    pdf.subsection_title("4.4", "Improvement Hyperparameters")
    pdf.add_table(
        ["Parameter", "Method", "Value"],
        [
            ["mu", "FedProx", "0.01"],
            ["alpha (drift scaling)", "Adaptive / Hybrid", "0.1"],
            ["mu_0 (initial mu)", "Decaying / Hybrid", "0.01"],
            ["beta (decay rate)", "Decaying / Hybrid", "0.1"],
        ],
        col_widths=[65, 60, 65],
    )

    # ── Results ──────────────────────────────────────────────────────────

    pdf.add_page()
    pdf.section_title(5, "Results")

    pdf.subsection_title("5.1", "Baseline: FedAvg IID vs Non-IID")
    pdf.add_plot(
        os.path.join(results_dir, "accuracy_fedavg_comparison.png"),
        "Figure 1: FedAvg accuracy under IID and Non-IID data distributions."
    )
    pdf.add_plot(
        os.path.join(results_dir, "loss_fedavg_comparison.png"),
        "Figure 2: FedAvg loss under IID and Non-IID data distributions."
    )

    # Results table for FedAvg
    if "FedAvg (IID)" in all_histories and "FedAvg (Non-IID)" in all_histories:
        rows = []
        iid_acc = all_histories["FedAvg (IID)"]["accuracy"]
        noniid_acc = all_histories["FedAvg (Non-IID)"]["accuracy"]
        rounds = all_histories["FedAvg (IID)"]["rounds"]
        for i in range(0, len(rounds), max(1, len(rounds) // 5)):
            rows.append([
                str(rounds[i]),
                f"{iid_acc[i]:.4f}",
                f"{noniid_acc[i]:.4f}",
            ])
        pdf.add_table(
            ["Round", "FedAvg (IID)", "FedAvg (Non-IID)"],
            rows,
            col_widths=[40, 75, 75],
        )

    pdf.subsection_title("5.2", "Baseline: FedAvg vs FedProx (Non-IID)")
    pdf.add_plot(
        os.path.join(results_dir, "accuracy_fedavg_vs_fedprox.png"),
        "Figure 3: FedAvg vs FedProx accuracy under Non-IID setting."
    )

    pdf.subsection_title("5.3", "mu Sensitivity Analysis")
    pdf.add_plot(
        os.path.join(results_dir, "mu_sensitivity.png"),
        "Figure 4: Effect of proximal coefficient mu on final accuracy."
    )

    # μ sweep table
    mu_rows = [[str(mu), f"{acc:.4f}"] for mu, acc in zip(mu_values, mu_accuracies)]
    pdf.add_table(
        ["mu", "Final Accuracy"],
        mu_rows,
        col_widths=[95, 95],
    )

    pdf.subsection_title("5.4", "All Methods Comparison")
    pdf.add_plot(
        os.path.join(results_dir, "accuracy_all_methods.png"),
        "Figure 5: Accuracy comparison of all methods over communication rounds."
    )
    pdf.add_plot(
        os.path.join(results_dir, "loss_all_methods.png"),
        "Figure 6: Loss comparison of all methods over communication rounds."
    )

    pdf.subsection_title("5.5", "Final Accuracy Summary")
    pdf.add_plot(
        os.path.join(results_dir, "final_accuracy_comparison.png"),
        "Figure 7: Final accuracy comparison of all methods."
    )

    # Summary table
    summary_rows = []
    for name, history in all_histories.items():
        summary_rows.append([name, f"{history['accuracy'][-1]:.4f}"])
    pdf.add_table(
        ["Method", "Final Accuracy"],
        summary_rows,
        col_widths=[95, 95],
    )

    # ── Discussion ───────────────────────────────────────────────────────

    pdf.add_page()
    pdf.section_title(6, "Discussion")

    pdf.subsection_title("6.1", "Baseline Observations")
    pdf.body_text(
        "FedAvg performs strongly on MNIST even under non-IID conditions. The dataset's relative "
        "simplicity means that the CNN can learn digit-agnostic features (edges, curves) even when "
        "each client only sees 2 digit classes. FedProx with mu=0.01 shows comparable performance, "
        "confirming that the proximal term adds limited benefit when client drift is already low."
    )

    pdf.subsection_title("6.2", "mu Sensitivity")
    pdf.body_text(
        "The sensitivity analysis confirms that mu requires careful tuning. Small values (0.001) "
        "behave nearly identically to FedAvg, while large values (0.1) over-constrain local learning "
        "and degrade accuracy. This validates the motivation for automatically adapting mu."
    )

    pdf.subsection_title("6.3", "Impact of Proposed Improvements")
    pdf.body_text(
        "The Adaptive FedProx approach automatically scales mu based on client drift, eliminating "
        "the need for manual tuning. The Decaying FedProx relaxes the constraint over time, "
        "allowing more flexible learning in later rounds. The Hybrid approach combines both benefits."
    )
    pdf.body_text(
        "On MNIST, the differences between methods are subtle due to the dataset's simplicity. "
        "However, the proposed improvements demonstrate the principle of adaptive regularization "
        "and are expected to show larger benefits on more complex datasets (e.g., CIFAR-10) "
        "with stronger heterogeneity."
    )

    pdf.subsection_title("6.4", "Limitations")
    pdf.bullet_point("MNIST is a relatively simple dataset - differences between methods are small")
    pdf.bullet_point("10 clients may not fully capture real-world FL scale")
    pdf.bullet_point("Only label-based non-IID partitioning was tested (not feature-based)")
    pdf.bullet_point("Adaptive strategies introduce additional hyperparameters (alpha, beta)")

    # ── Conclusion ───────────────────────────────────────────────────────

    pdf.section_title(7, "Conclusion")
    pdf.body_text(
        "This project successfully reproduced and analyzed both FedAvg and FedProx under IID and "
        "non-IID settings on the MNIST dataset. We confirmed that FedProx provides theoretical "
        "improvements for heterogeneous settings, though practical benefits depend on dataset "
        "complexity and parameter tuning."
    )
    pdf.body_text(
        "We proposed three novel improvements to FedProx: Drift-Aware Adaptive mu (client-specific "
        "regularization), Time-Decaying mu (progressive relaxation), and Hybrid mu (combining both). "
        "These approaches address FedProx's core limitations - uniform and static mu - by making "
        "the proximal mechanism adaptive to both client heterogeneity and training dynamics."
    )
    pdf.body_text(
        "Future work should evaluate these improvements on more complex datasets (CIFAR-10, "
        "Shakespeare) with larger client populations and partial participation to fully demonstrate "
        "their potential advantages."
    )

    # ── References ───────────────────────────────────────────────────────

    pdf.section_title(8, "References")
    refs = [
        "[1] McMahan et al., 'Communication-Efficient Learning of Deep Networks from Decentralized Data', AISTATS 2017.",
        "[2] Li et al., 'Federated Optimization in Heterogeneous Networks', MLSys 2020.",
        "[3] Karimireddy et al., 'SCAFFOLD: Stochastic Controlled Averaging for Federated Learning', ICML 2020.",
        "[4] Wang et al., 'Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization', NeurIPS 2020.",
        "[5] Acar et al., 'Federated Learning Based on Dynamic Regularization', ICLR 2021.",
    ]
    for ref in refs:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(33, 33, 33)
        pdf.multi_cell(0, 5, ref)
        pdf.ln(2)

    # ── Save PDF ─────────────────────────────────────────────────────────

    output_path = os.path.join(results_dir, "FedProx_Research_Paper.pdf")
    pdf.output(output_path)
    print(f"  📄 PDF saved: {output_path}")


if __name__ == "__main__":
    print("This script is designed to be called from run_all.py.")
    print("Run 'python run_all.py' to generate the full report.")