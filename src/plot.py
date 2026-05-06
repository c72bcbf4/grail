import os
from datetime import datetime

import pandas as pd
from matplotlib import pyplot


def plot_training(logs, times, directory, overrides):
    data = pd.DataFrame(logs).round(6)

    data.to_pickle(os.path.join(directory, "data.pkl"))

    plots = [(data, "all")]

    for df, suffix in plots:
        file = os.path.join(directory, f"training_{suffix}")

        epoch = logs[-1]["epoch"]
        y = df["step"]

        fig, axes = pyplot.subplots(nrows=5, ncols=3, figsize=(16, 12))

        axes = axes.T

        losses_train = df.filter(like="loss").drop(columns=["eval_loss"])
        losses_eval = df[["eval_loss"]]

        grad_norm = df.filter(like="grad_norm")

        metrics = df[
            ["f1_graph", "f1_edges", "tp_graph", "tn_graph", "fp_graph", "fn_graph"]
        ]

        lr = df.filter(like="lr")

        corrects = df[["correct"]]

        param_norms = df.filter(like="param_norm")
        durations = df.filter(like="duration_")
        system = df.filter(like="sys_")

        logits = df.filter(like="logits")

        lens = df.filter(like="length_")
        successors_sum = df[["successors_sum"]]
        successors_avg = df[["successors_avg"]]

        top_k = df[["top_k", "pos_frac_graph", "pos_frac_edges"]]

        plots = [
            (losses_train, 0, 0, None, None),
            (losses_eval, 0, 1, None, None),
            (logits, 0, 2, None, None),
            (metrics, 0, 3, None, None),
            (times, 0, 4, "times", None),
            (corrects, 1, 0, None, None),
            (lens, 1, 1, None, None),
            (top_k, 1, 2, None, None),
            (successors_avg, 1, 3, None, None),
            (successors_sum, 1, 4, None, None),
            (durations, 2, 0, None, None),
            (lr, 2, 1, None, None),
            (grad_norm, 2, 2, None, None),
            (param_norms, 2, 3, None, None),
            (system, 2, 4, None, None),
        ]

        for i, (data, row, col, extra, label) in enumerate(plots):
            ax = axes[row, col]
            if data is None:
                ax.axis("off")
                continue

            if extra == "times":
                keys = list(times.keys())
                labels = [f"{k/100}%" for k in keys]
                values = [times[k] for k in keys]

                for p in [10**0, 10**1, 10**2, 10**3]:
                    ax.axhline(p, color="gray", linestyle="--", linewidth=0.8)

                ax.bar(labels, values, color="skyblue", edgecolor="navy")
                ax.tick_params(axis="x", labelrotation=90)

                ax.set_xlabel("Quantiles")
                ax.set_ylabel("Time (ms)")
                ax.set_yscale("symlog")
                continue

            label = [f"{k} ({v})" for k, v in data.iloc[-1].to_dict().items()]
            ax.plot(y, data, label=label)
            ax.legend(loc="upper left")
            ax.grid()

        groups = {}
        for o in overrides:
            key, value = o.split("=")
            key = key.split("@")[-1]
            if "." in key:
                parts = key.split(".")
                group = ".".join(parts[:-1])
                key = parts[-1]

                groups.setdefault(group, [])
                groups[group].append(f"{key}={value}")
            else:
                groups.setdefault(key, [])
                groups[key].append(f"{value}")

        n = 5
        rows = []
        for k, v in groups.items():
            for i in range(0, len(v), n):
                rows.append((k, v[i : i + n]))

        rows = [f"{k}: " + ", ".join(v) for k, v in rows]

        title = "\n".join(rows)

        date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

        fig.suptitle(f"epoch {epoch} {date}\n{title}")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - len(groups) * 0.005))
        fig.savefig(file)
        pyplot.close(fig)
