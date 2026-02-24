import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import ceil, floor, sqrt, pi, exp
import matplotlib.ticker as mtick


# ==================== USER INPUTS ====================
BASE_DIR = Path(__file__).resolve().parent          # folder this script is in
CSV_PATH = BASE_DIR / "loot_table_template.csv"     # your filled CSV
COST_PER_PLAY = 665                                 # x: cost per roll
NS = [8]                                            # n values to graph
BIN_SIZE = 20.0                                     # USD bin width (e.g., 1.00, 0.10, 0.01)
OUTPUT_DIR = BASE_DIR / "outputs"                   # output folder
OVERLAY_NORMAL = True                               # overlay Normal approx when n >= 20
# =====================================================

def load_table(csv_path):
    df = pd.read_csv(csv_path)
    req = {"Item","ValueUSD","Probability"}
    if not req.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {req}")
    if (df["Probability"] < 0).any():
        raise ValueError("Probabilities must be non-negative")
    s = df["Probability"].sum()
    if abs(s - 1.0) > 1e-9:
        print(f"[note] Probabilities sum to {s:.12f}, renormalizing to 1.")
        df["Probability"] = df["Probability"] / s
    return df

def moments(values, probs):
    mu = np.sum(values * probs)
    var = np.sum((values - mu)**2 * probs)
    return mu, var

def build_single_play_pmf_binned(df, cost, bin_size):
    # per-play profit
    profits = df["ValueUSD"].to_numpy() - float(cost)
    probs   = df["Probability"].to_numpy()

    # Bin to grid of size bin_size (round to nearest bin center)
    scale = 1.0 / float(bin_size)
    # Convert to integer bins
    int_bins = np.rint(profits * scale).astype(int)
    min_b = int_bins.min(); max_b = int_bins.max()
    width = max_b - min_b + 1
    pmf = np.zeros(width, dtype=np.float64)
    for b, p in zip(int_bins, probs):
        pmf[b - min_b] += p
    x_vals = (np.arange(width) + min_b) / scale
    return x_vals, pmf, scale

def n_fold_convolution_fft(pmf, n):
    if n == 1:
        return pmf.copy()
    target_len = (len(pmf)-1)*n + 1
    # next power of two for FFT length
    fft_len = 1
    while fft_len < target_len:
        fft_len <<= 1
    P = np.fft.rfft(np.pad(pmf, (0, fft_len-len(pmf)), mode='constant'))
    conv = np.fft.irfft(P**n, n=fft_len)[:target_len]
    conv[conv < 0] = 0
    s = conv.sum()
    if s > 0:
        conv /= s
    return conv

def normal_pdf(x, mean, std):
    return (1.0 / (std * sqrt(2*pi))) * np.exp(-0.5 * ((x-mean)/std)**2)

def main():
    outdir = Path(OUTPUT_DIR); outdir.mkdir(parents=True, exist_ok=True)

    df = load_table(CSV_PATH)
    values = df["ValueUSD"].to_numpy() - float(COST_PER_PLAY)
    probs  = df["Probability"].to_numpy()
    mu1, var1 = moments(values, probs)
    sd1 = sqrt(var1)

    x1, pmf1, scale = build_single_play_pmf_binned(df, COST_PER_PLAY, BIN_SIZE)

    print(f"Per-play EV: {mu1:.4f} USD | SD: {sd1:.4f} USD | Bin size: ${BIN_SIZE:.2f}")
    all_rows = []

    for n in NS:
        pmfn = n_fold_convolution_fft(pmf1, n)
        step = BIN_SIZE
        cdf = np.cumsum(pmfn)
        x_min = x1.min() * n
        x_max = x1.max() * n
        x_vals = np.arange(x_min, x_max + step/2, step)
        if len(x_vals) != len(pmfn):
            x_vals = x_vals[:len(pmfn)]

        evn = np.sum(x_vals * pmfn)
        pr_win = pmfn[x_vals > 0].sum()
        # Median total profit: point where CDF crosses 0.5
        median_profit = np.interp(0.5, cdf, x_vals)

        print(f"n={n:>4} | EV={evn:.2f} | P(total profit > 0)={pr_win:.4%} | support={len(x_vals)} bins")
        total_invested = n * float(COST_PER_PLAY)
        # Plot
        import matplotlib.pyplot as plt
        plt.figure()
        ev_return = (df["ValueUSD"] * df["Probability"]).sum()
        ev_profit = ev_return - COST_PER_PLAY
        print(f"Expected return per play: ${ev_return:.2f}")
        print(f"Expected profit per play: ${ev_profit:.2f}")
        print(f"Total Expected return   : ${ev_return*n:.2f}")
        print(f"Total Expected profit   : ${ev_profit*n:.2f}")
        plt.title(
        f"Profit Distribution for n = {n}\n"
        f"Total invested: ${total_invested:,.2f} | "
        f"EV (profit): ${evn:,.2f} | "
        f"P(Profit > 0): {pr_win:.2%}"
        )
        plt.xlabel("Total Profit ($)")
        plt.ylabel(f"Probability per ${BIN_SIZE:.2f} bin and cost per play{COST_PER_PLAY:.0f}")
        plt.plot(x_vals, pmfn, lw=1.2, label="Exact (FFT)")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"${x:,.0f}"))
        # Break-even vertical line at $0
        plt.axvline(0, linestyle="--", linewidth=1.2, color="red", label="Break-even")
        # Mean (expected profit)
        plt.axvline(evn, linestyle="-.", linewidth=1.2, color="purple", label="Mean (EV)")
        # Median (50% of area under curve)
        plt.axvline(median_profit, linestyle=":", linewidth=1.2, color="green", label="Median (50% area)")
        # choose how much of the tail to keep; 0.001 = 0.1% on each side
        low_q  = 0.01
        high_q = 0.990
        x_low  = np.interp(low_q,  cdf, x_vals)
        x_high = np.interp(high_q, cdf, x_vals)
        plt.xlim(x_low, x_high)
        if OVERLAY_NORMAL and n >= 20 and var1 > 0:
            mean_n = n * mu1
            std_n  = sqrt(n * var1)
            # sample normal over same x grid for visual overlay
            pdf = normal_pdf(x_vals, mean_n, std_n)
            # normalize to area ~1 over grid step
            pdf_discrete = pdf * step
            plt.plot(x_vals, pdf_discrete, lw=1.0, linestyle="--", label="Normal approx")

        plt.legend()
        plt.tight_layout()
        png_path = outdir / f"profit_pmf_n{n}.png"
        plt.savefig(png_path, dpi=150)
        plt.close()

        # Save CSV for this n
        pd.DataFrame({"TotalProfitUSD": x_vals, "Probability": pmfn}).to_csv(
            outdir / f"profit_pmf_n{n}.csv", index=False
        )

        for xv, pv in zip(x_vals, pmfn):
            all_rows.append({"n": n, "TotalProfitUSD": xv, "Probability": pv})

    pd.DataFrame(all_rows).to_csv(outdir / "all_pmfs_long.csv", index=False)
    # Also save a small summary file
    summary = pd.DataFrame({"NS": NS})
    summary.to_csv(outdir / "summary_ns.csv", index=False)
    print(f"Outputs saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
