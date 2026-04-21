"""Find which GPU min_variance configs disagree with v3."""
import numpy as np
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def main():
    from src.data_loader import load_returns
    from src.backtest import default_config_grid
    from src.versions import v3_numba, v5_gpu

    returns = load_returns().to_numpy()
    configs = default_config_grid()
    n_bootstrap = 10

    # Just min_variance
    minvar_configs = [c for c in configs if c.objective == "min_variance"]
    print(f"Running {len(minvar_configs)} min_variance configs x {n_bootstrap} bootstrap")

    print("\nRunning v3...")
    v3r = v3_numba.run(returns, minvar_configs, n_bootstrap=n_bootstrap,
                        seed=42, show_progress=False)

    print("Running v5...")
    v5r = v5_gpu.run(returns, minvar_configs, n_bootstrap=n_bootstrap,
                      seed=42, show_progress=False)

    df3 = v3r.raw_metrics.sort_values(["config_idx", "bootstrap_idx"]).reset_index(drop=True)
    df5 = v5r.raw_metrics.sort_values(["config_idx", "bootstrap_idx"]).reset_index(drop=True)

    print(f"\n{'cfg':>3s} {'label':<50s} {'max_diff':>10s} {'v3 mean':>10s} {'v5 mean':>10s}")
    print("-" * 90)
    for cfg_idx, cfg in enumerate(minvar_configs):
        s3 = df3[df3["config_idx"] == cfg_idx]["sharpe"].to_numpy()
        s5 = df5[df5["config_idx"] == cfg_idx]["sharpe"].to_numpy()
        d = np.abs(s3 - s5).max()
        flag = " <--" if d > 1e-3 else ""
        print(f"{cfg_idx:>3d} {cfg.label():<50s} {d:>10.2e} "
              f"{s3.mean():>10.4f} {s5.mean():>10.4f}{flag}")


if __name__ == "__main__":
    main()