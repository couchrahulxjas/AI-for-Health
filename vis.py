# imports

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages




def load_signal(path):

    with open(path, "r", encoding="utf-8", errors="replace") as f:                  # loading path
        lines = f.readlines()

    
    data_start = 0                                                                  # # Find data start — after 'Data:' header if present
    for i, line in enumerate(lines):
        if line.strip() == "Data:":
            data_start = i + 1
            break

    timestamps, values = [], []

    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split(";")                                                     # If the line does not contain at least two fields, skip it
        if len(parts) < 2:
            continue

        try:
            raw_time = parts[0].strip()                                             # [::-1] reverses string  
            value = float(parts[1].strip().replace(",", "."))                       # replace(",", ".", 1) replaces last comma with dot (for milliseconds)
            ts = None                                                               # Initialize timestamp variable
            time_str = raw_time[::-1].replace(",", ".", 1)[::-1]
            value    = float(parts[1].strip().replace(",", "."))

            ts = None
            for fmt in (                                                    # as dada had various timestamp formats, we try multiple until one works
                "%d.%m.%Y %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S.%f",
                "%d.%m.%Y %H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    ts = pd.to_datetime(time_str, format=fmt)
                    break
                except ValueError:
                    pass

            if ts is not None:
                timestamps.append(ts)
                values.append(value)
        except Exception:
            continue

    return np.array(timestamps), np.array(values, dtype=np.float64)










def load_events(path):

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    starts, ends, labels = [], [], []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split(";")
        if len(parts) < 2:
            continue


        time_range = parts[0].strip()


        if not (len(time_range) > 10 and time_range[2] == "." and time_range[5] == "."):            #  Data lines start with DD.MM.YYYY so check for dots at positions 2 and 5
            continue
        if "-" not in time_range:                                                                   # if not present skip.
            continue

        label = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else "Event"

        try:
            space_idx = time_range.index(" ")                                                   # Find space separating date and time
            date_part = time_range[:space_idx]                                                  # extracting date part
            time_part = time_range[space_idx + 1:]                                              # extracting time part.


            dash_idx  = time_part.rfind("-")                                                    # Find last dash to split start and end time
            s_str     = time_part[:dash_idx].replace(",", ".")                                  # replace
            e_str     = time_part[dash_idx + 1:].replace(",", ".")  


            start = pd.to_datetime(f"{date_part} {s_str}", format="%d.%m.%Y %H:%M:%S.%f")
            end   = pd.to_datetime(f"{date_part} {e_str}", format="%d.%m.%Y %H:%M:%S.%f")

            
            if end < start:
                end += pd.Timedelta(days=1)                                                     # Handle midnight crossing by adding one day to end time if it is earlier than start time.

            starts.append(start)
            ends.append(end)
            labels.append(label)

        except Exception:
            continue

    print(f"  Events loaded : {len(starts)}")
    return np.array(starts), np.array(ends), np.array(labels)









def find_files(folder):
    files = os.listdir(folder)

    def pick(keywords, exclude=None):                                           # pick a file based on keywords
        exclude = exclude or []


        for f in sorted(files):
            fl = f.lower()
            if any(ex in fl for ex in exclude):                                 # skip if any exclude keyword is present  
                continue
            if any(kw in fl for kw in keywords):                                # if any keyword is present, return full path
                return os.path.join(folder, f)
        return None
    

    airflow  = pick(["nasal", "airflow", "flow"], exclude=["event"])
    thoracic = pick(["thorac", "thorax", "chest"])
    spo2     = pick(["spo2", "spo₂", "oxygen", "o2"])
    events   = pick(["event"])


    missing = [k for k, v in [("airflow",  airflow),
                               ("thoracic", thoracic),
                               ("spo2",     spo2),
                               ("events",   events)] if v is None]
    if missing:
        raise FileNotFoundError(
            f"Could not find files for: {missing}\n"
            f"Available: {files}"
        )

    return airflow, thoracic, spo2, events




def get_color(label):
    label = label.lower()
    if "obstructive" in label: return "#e74c3c"
    if "hypopnea"    in label: return "#e67e22"
    if "central"     in label: return "#9b59b6"
    if "mixed"       in label: return "#1abc9c"
    return "#e74c3c"




def visualize(name, data_dir="Data", out_dir="Visualizations"):
    folder = os.path.join(data_dir, name)

    
    air_file, thor_file, spo2_file, event_file = find_files(folder)
    ts_air,  airflow  = load_signal(air_file)                                   # load airflow signal
    ts_thor, thoracic = load_signal(thor_file)                                  # load thoracic signal                   
    ts_spo2, spo2     = load_signal(spo2_file)                                  # load spo2 signal          
    ev_starts, ev_ends, ev_labels = load_events(event_file)                     # load events






    fig, axes = plt.subplots(3, 1, figsize=(22, 12), sharex=True)
    fig.suptitle(f"Sleep Recording — {name}", fontsize=15,fontweight="bold")

   
    axes[0].plot(ts_air,  airflow,  color="#2471a3", linewidth=0.5, alpha=0.85)                     #  x-axis (timestamps),  Line color (blue tone)
    axes[0].set_title("Nasal Airflow",     fontsize=11, fontweight="bold", loc="left")                # y-axis (airflow signal values)
    axes[0].set_ylabel("Signal (a.u.)",    fontsize=9)
    axes[0].grid(alpha=0.25, linestyle="--")

    axes[1].plot(ts_thor, thoracic, color="#1e8449", linewidth=0.5, alpha=0.85)
    axes[1].set_title("Thoracic Movement", fontsize=11, fontweight="bold", loc="left")
    axes[1].set_ylabel("Signal (a.u.)",    fontsize=9)
    axes[1].grid(alpha=0.25, linestyle="--")

    axes[2].plot(ts_spo2, spo2,     color="#7d3c98", linewidth=1.0, alpha=0.85)
    axes[2].set_title("SpO₂ (%)",          fontsize=11, fontweight="bold", loc="left")
    axes[2].set_ylabel("SpO₂ (%)",         fontsize=9)
    axes[2].grid(alpha=0.25, linestyle="--")







    colors_used = {}
    for start, end, label in zip(ev_starts, ev_ends, ev_labels):                        # Event start time
        color = get_color(label)                                                         # Event end time
        for ax in axes:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),                  
                       color=color, alpha=0.30, linewidth=0)                            #         # Transparency (light shading)
        if label not in colors_used:
            colors_used[label] = color






    # Add legend

    if colors_used:
        handles = [
            mpatches.Patch(facecolor=c, alpha=0.6, label=l)
            for l, c in colors_used.items()
        ]
        axes[0].legend(handles=handles, loc="upper right",
                       fontsize=8, title="Annotated Events", title_fontsize=8)
    else:
        axes[0].text(0.99, 0.95, "No events found",
                     transform=axes[0].transAxes,
                     ha="right", va="top", fontsize=8, color="gray")









    # Save PDF 
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{name}_visualization.pdf")

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
        d = pdf.infodict()
        d["Title"]   = f"Sleep Recording — {name}"
        d["Subject"] = "Physiological signals with annotated breathing events"

    plt.close(fig)
    print(f"\n  PDF saved → {pdf_path}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize sleep signals")
    parser.add_argument("-name",     required=True, help="Participant ID e.g. AP01")
    parser.add_argument("-data_dir", default="Data", help="Data folder (default: Data)")
    parser.add_argument("-out_dir",  default="Visualizations", help="Output folder")
    args = parser.parse_args()

    visualize(args.name, args.data_dir, args.out_dir)