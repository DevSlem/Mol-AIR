import argparse
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from util import try_create_dir, moving_average, load_yaml

def make_plot_figure(x_vals_list, y_vals_list, labels, x_label, y_label, ylim=None):
    fig, ax = plt.subplots()
    
    for x_vals, y_vals, label in zip(x_vals_list, y_vals_list, labels):
        ax.plot(x_vals, y_vals, label=label)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    if ylim is not None:
        ax.set_ylim(top=ylim)
    
    return fig, ax

def df_groupby_episode(df_list, func):
    df_result_list = [df.groupby("episode") for df in df_list]
    df_result_list = [func(df) for df in df_result_list]
    df_result_list = [df[~df.isna()] for df in df_result_list]
    return df_result_list

def series_to_xy(series_list):
    x_vals_list = [series.index for series in series_list]
    y_vals_list = [series.values for series in series_list]
    return x_vals_list, y_vals_list

def avg_score_figure(episode_metric_df_list, title, labels, n):
    avg_score_list = df_groupby_episode(
        episode_metric_df_list,
        lambda x: x["score"].mean()
    )
    x_vals_list, y_vals_list = series_to_xy(avg_score_list)
    avg_score_fig, _ = make_plot_figure(
        x_vals_list,
        y_vals_list,
        labels,
        "Episode",
        f"Average {title}"
    )
    moving_avg_score_fig, _ = make_plot_figure(
        x_vals_list,
        [moving_average(y_vals, n=n) for y_vals in y_vals_list],
        labels,
        "Episode",
        f"Moving Average {title} (n: {n})"
    )
    return avg_score_fig, moving_avg_score_fig
    
def anoot_val(x, y, ax=None):
    text= f"y={y:.3f}"
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(xycoords='data',textcoords="axes fraction",
              bbox=bbox_props, ha="right", va="bottom")
    ax.annotate(text, xy=(x, y), xytext=(0.94,0.96), **kw)


def best_score_figure(episode_metrics, title, labels):
    fig, ax = plt.subplots()
    
    xmax = -1
    ymax = -100000
    
    for df, label in zip(episode_metrics, labels):
        best_scores = df.groupby("episode")["score"].max()
        best_scores = best_scores[~best_scores.isna()]
        curruent_best_score_episode = best_scores.index[0]
        current_best_score = best_scores[curruent_best_score_episode]
        for episode in best_scores.index:
            if best_scores[episode] > current_best_score:
                current_best_score = best_scores[episode]
                curruent_best_score_episode = episode
            best_scores[episode] = current_best_score
        ax.plot(best_scores, label=label)
        
        if current_best_score > ymax:
            xmax = curruent_best_score_episode
            ymax = current_best_score
        
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Best {title}")
    ax.legend()
    anoot_val(xmax, ymax, ax)
    
    return fig

def avg_int_reward_figure(episode_metric_df_list, title, labels, int_reward_type, n):
    if int_reward_type.lower() == "count":
        column_field = "avg_count_int_reward"
    elif int_reward_type.lower() == "rnd":
        column_field = "avg_rnd_int_reward"
    else:
        raise ValueError(f"Invalid int_reward_type: {int_reward_type}")
    
    filtered_df = []
    filtered_labels = []
    for df, label in zip(episode_metric_df_list, labels):
        if column_field in df.columns:
            filtered_df.append(df)
            filtered_labels.append(label)
            
    if len(filtered_df) == 0:
        return None
            
    avg_int_rewards_list = df_groupby_episode(
        filtered_df,
        lambda x: x[column_field].mean()
    )
    x_vals_list, y_vals_list = series_to_xy(avg_int_rewards_list)
    fig, _ = make_plot_figure(
        x_vals_list,
        y_vals_list,
        filtered_labels,
        "Episode",
        f"{title} Average {int_reward_type} Intrinsic Reward"
    )
    ma_fig, _ = make_plot_figure(
        x_vals_list,
        [moving_average(y_vals, n=n) for y_vals in y_vals_list],
        filtered_labels,
        "Episode",
        f"{title} Moving Average {int_reward_type} Intrinsic Reward (n: {n})"
    )
    return fig, ma_fig

def total_avg_int_reward_figure(
    episode_metric_df_list,
    title,
    labels,
    n,
    count_coefs,
    rnd_coefs
):  
    filtered_count_df = {}
    for df, label in zip(episode_metric_df_list, labels):
        if "avg_count_int_reward" in df.columns:
            filtered_count_df[label] = df
            
    if len(filtered_count_df) != len(count_coefs):
        raise ValueError("The number of Count coefficients must be equal to the number of Count int rewards.")
    
    filtered_rnd_df = {}
    for df, label in zip(episode_metric_df_list, labels):
        if "avg_rnd_int_reward" in df.columns:
            filtered_rnd_df[label] = df
            
    if len(filtered_rnd_df) != len(rnd_coefs):
        raise ValueError("The number of RND coefficients must be equal to the number of RND int rewards.")
            
    if len(filtered_count_df) == 0 and len(filtered_rnd_df) == 0:
        return None
    
    avg_count_int_reward_list = df_groupby_episode(
        filtered_count_df.values(),
        lambda x: x["avg_count_int_reward"].mean()
    )
    avg_rnd_int_rewards_list = df_groupby_episode(
        filtered_rnd_df.values(),
        lambda x: x["avg_rnd_int_reward"].mean()
    )
    avg_count_int_reward_list = [count_coefs[i] * avg_count_int_reward_list[i] for i in range(len(avg_count_int_reward_list))]
    avg_rnd_int_rewards_list = [rnd_coefs[i] * avg_rnd_int_rewards_list[i] for i in range(len(avg_rnd_int_rewards_list))]
    
    total_avg_int_rewards_dict = {}
    for i, label in enumerate(filtered_count_df.keys()):
        total_avg_int_rewards_dict[label] = avg_count_int_reward_list[i]
    for i, label in enumerate(filtered_rnd_df.keys()):
        if label in total_avg_int_rewards_dict:
            total_avg_int_rewards_dict[label] += avg_rnd_int_rewards_list[i]
        else:
            total_avg_int_rewards_dict[label] = avg_rnd_int_rewards_list[i]
    
    avg_count_int_reward_means = [avg_count_int_reward.mean() for avg_count_int_reward in avg_count_int_reward_list]
    avg_rnd_int_reward_means = [avg_rnd_int_reward.mean() for avg_rnd_int_reward in avg_rnd_int_rewards_list]
    # make a barplot
    count_color = "blue"
    rnd_color = "green"
    barfig, barax = plt.subplots()
    bottoms = {}
    for i, label in enumerate(filtered_count_df.keys()):
        barax.bar(label, avg_count_int_reward_means[i], color=count_color)
        bottoms[label] = avg_count_int_reward_means[i]
    for i, label in enumerate(filtered_rnd_df.keys()):
        bottom = bottoms[label] if label in bottoms else 0
        barax.bar(label, avg_rnd_int_reward_means[i], bottom=bottom, color=rnd_color)
    legend_elements = [
        Patch(facecolor=count_color, label="Count"),
        Patch(facecolor=rnd_color, label="RND")
    ]
    barax.set_title(f"{title} Intrinsic Reward Mean Bar by Type")
    barax.set_xlabel("Labels")
    barax.set_ylabel("Mean")
    # barax.set_yscale("log")
    barax.legend(handles=legend_elements)
    
    
    x_vals_list, y_vals_list = series_to_xy(total_avg_int_rewards_dict.values())
    total_y_vals = np.concatenate(y_vals_list)
    total_mean = total_y_vals.mean()
    total_std = total_y_vals.std()
    ylim = total_mean + 3 * total_std
    fig, _ = make_plot_figure(
        x_vals_list,
        y_vals_list,
        total_avg_int_rewards_dict.keys(),
        "Episode",
        f"{title} Average Intrinsic Reward",
        ylim
    )
    ma_fig, _ = make_plot_figure(
        x_vals_list,
        [moving_average(y_vals, n=n) for y_vals in y_vals_list],
        total_avg_int_rewards_dict.keys(),
        "Episode",
        f"{title} Moving Average Intrinsic Reward (n: {n})",
        ylim
    )
    return fig, ma_fig, barfig

def best_comparison_table(episode_metrics, labels) -> pd.DataFrame:    
    comparison_df = pd.DataFrame(
        columns=["Label", "Episode", "Score", "Selfies"]
    )
    
    for i, df in enumerate(episode_metrics):
        best_row = df.iloc[df["score"].argmax()]
        comparison_df.loc[i] = [labels[i], best_row["episode"], best_row["score"], best_row["selfies"]]
    
    return comparison_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('TITLE', type=str, help='Title for the data')
    parser.add_argument('EXPERIMENT_RESULT_DIR', nargs='+', type=str, help='Directories of experiment results')
    parser.add_argument('-e', '--episode', type=int, help='Episode number (default = max)', default=None)
    parser.add_argument('-m', '--moving_average', type=int, help='Moving average n (default = max_episode / 100)', default=None)
    
    args = parser.parse_args()

    title = args.TITLE
    experiment_result_dirs = args.EXPERIMENT_RESULT_DIR
    episode = args.episode
    moving_average_n = args.moving_average
    
    labels = []
    count_coefs = []
    rnd_coefs = []
    for d in experiment_result_dirs:
        config_dict = load_yaml(f"{d}/config.yaml")
        label = tuple(config_dict.keys())[0]
        labels.append(label)
        config_dict = config_dict[label]
        if "CountIntReward" in config_dict and "crwd_coef" in config_dict["CountIntReward"]:
            count_coefs.append(config_dict["CountIntReward"]["crwd_coef"])
        if "nonepi_adv_coef" in config_dict["Agent"]:
            rnd_coefs.append(config_dict["Agent"]["nonepi_adv_coef"])
    
    episode_metrics = []
    for d in experiment_result_dirs:
        episode_metrics.append(pd.read_csv(f"{d}/episode_metric.csv"))
    
    if episode is None:
        episode = min([df["episode"].max() for df in episode_metrics])
    
    if moving_average_n is None:
        moving_average_n = episode // 100
    
    episode_metrics = [df[df["episode"] <= episode] for df in episode_metrics]
    episode_metrics = [df.sort_values(by=["episode", "env_id"]) for df in episode_metrics]

    avg_score_fig, moving_avg_score_fig = avg_score_figure(episode_metrics, title, labels, moving_average_n)
    best_score_fig = best_score_figure(episode_metrics, title, labels)
    avg_count_int_reward_figs = avg_int_reward_figure(episode_metrics, title, labels, "Count", moving_average_n)
    avg_rnd_int_reward_figs = avg_int_reward_figure(episode_metrics, title, labels, "RND", moving_average_n)
    avg_int_reward_figs = total_avg_int_reward_figure(
        episode_metrics,
        title,
        labels,
        moving_average_n,
        count_coefs,
        rnd_coefs
    )
    best_comparison_df = best_comparison_table(episode_metrics, labels)
    
    export_dir = f"exports/{title}"
    
    try_create_dir(export_dir)
    
    avg_score_fig.savefig(f"{export_dir}/avg_score.png")
    moving_avg_score_fig.savefig(f"{export_dir}/moving_avg_score.png")
    best_score_fig.savefig(f"{export_dir}/best_score.png")
    if avg_count_int_reward_figs is not None:
        avg_count_int_reward_figs[0].savefig(f"{export_dir}/avg_count_int_reward.png")
        avg_count_int_reward_figs[1].savefig(f"{export_dir}/moving_avg_count_int_reward.png")
    if avg_rnd_int_reward_figs is not None:
        avg_rnd_int_reward_figs[0].savefig(f"{export_dir}/avg_rnd_int_reward.png")
        avg_rnd_int_reward_figs[1].savefig(f"{export_dir}/moving_avg_rnd_int_reward.png")
    if avg_int_reward_figs is not None:
        avg_int_reward_figs[0].savefig(f"{export_dir}/avg_int_reward.png")
        avg_int_reward_figs[1].savefig(f"{export_dir}/moving_avg_int_reward.png")
        avg_int_reward_figs[2].savefig(f"{export_dir}/avg_int_reward_bar.png")
    best_comparison_df.to_csv(f"{export_dir}/best_comparison.csv", index=False)
    
    print(f"Exported to {export_dir}")