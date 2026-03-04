# imports

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def load_signal(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    data_start = 0                                                         # get data line in this
    for i, line in enumerate(lines):
        if line.strip() == "Data:":
            data_start = i + 1
            break


    
    timestamps, values = [], []                                             # Lists to store timestamps and signal values

 
 
    for line in lines[data_start:]:
        line = line.strip()                                                 # strip() removes extra spaces and newline characters
        if not line:
            continue


        parts = line.split(";")                                         # # If line does not contain both timestamp and value, skip it
        if len(parts) < 2:
            continue
        try:
            time_str = parts[0].strip()[::-1].replace(",", ".", 1)[::-1]                # [::-1] reverses string  
            value = float(parts[1].strip().replace(",", "."))                           # replace(",", ".", 1) replaces last comma with dot (for milliseconds)
            ts = None                                                                    # Initialize timestamp variable
            for fmt in ("%d.%m.%Y %H:%M:%S.%f",
                         "%Y-%m-%d %H:%M:%S.%f",
                        "%d.%m.%Y %H:%M:%S",
                         "%Y-%m-%d %H:%M:%S"):
                


                try:
                    ts = pd.to_datetime(time_str, format=fmt)
                    break
                except ValueError:
                    pass
            if ts is not None:
                timestamps.append(ts)                                       # store timestamp and value in lists
                values.append(value)
        except Exception:
            continue

    return np.array(timestamps), np.array(values, dtype=np.float64)






# load events same as in vis.py.

def load_events(path):
    starts, ends, labels = [], [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(";")
            time_range = parts[0]
            label = parts[2] if len(parts) > 2 else "Unknown"

            date_part, time_part = time_range.split(" ")
            start_str, end_str = time_part.split("-")

            start = pd.to_datetime(f"{date_part} {start_str.replace(',', '.')}",    format="%d.%m.%Y %H:%M:%S.%f")
            end = pd.to_datetime(f"{date_part} {end_str.replace(',', '.')}",        format="%d.%m.%Y %H:%M:%S.%f")

            if end < start:
                end += pd.Timedelta(days=1)

            starts.append(start)
            ends.append(end)
            labels.append(label)

    return np.array(starts), np.array(ends), np.array(labels)





# find files in participant folder based on keywords in filename ( same as did in vis.py).

def find_files(participant_path):
    files = os.listdir(participant_path)

    def pick(keywords):
        for f in files:
            fl = f.lower()
            if any(kw in fl for kw in keywords):
                return os.path.join(participant_path, f)
        return None

    airflow  = pick(["nasal", "airflow", "flow"])
    thoracic = pick(["thorac", "thorax", "chest"])
    events   = pick(["event"])

    if not airflow or not thoracic or not events:
        raise FileNotFoundError("Required files not found")

    return airflow, thoracic, events






def bandpass_filter(signal, fs=32, lowcut=0.17, highcut=0.4, order=4):
    nyquist = 0.5 * fs                                                              # Calculate Nyquist frequency (half of sampling rate) and Used to normalize filter cutoff frequencies
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")       # Keeps frequencies between lowcut and highcut
    padlen = 3 * max(len(b), len(a))                                                # Minimum padding length required for filtfilt()
    if len(signal) <= padlen:
        return signal
    return filtfilt(b, a, signal)






def assign_label(window_start, window_end, ev_starts, ev_ends, ev_labels):
    window_duration   = (window_end - window_start).total_seconds()                         # Calculate duration of the window
    overlap_threshold = 0.5 * window_duration                                               #  Define threshold for significant overlap ( 50% of window duration as given in question)    

    for i in range(len(ev_starts)):
        ev_s = pd.Timestamp(ev_starts[i])
        ev_e = pd.Timestamp(ev_ends[i])
        if window_start < ev_e and window_end > ev_s:
            overlap = (min(window_end, ev_e) - max(window_start, ev_s)).total_seconds()     # If overlap exceeds threshold, assign that event label
            if overlap > overlap_threshold:
                return str(ev_labels[i])

    return "Normal"                                                                         # If no event overlaps enough then label window as Normal









def create_windows(airflow_f, thoracic_f, timestamps, ev_starts, ev_ends, ev_labels,  fs=32, window_sec=30, step_sec=15):                                          # Split the signals into 30-second windows with 50% overlap.(2 marks)
    window_samples = window_sec * fs                                                                                                                               # 30 sec × 32 Hz = 960 samples
    step_samples   = step_sec   * fs

    windows, labels = [], []

    for start_idx in range(0, len(airflow_f) - window_samples + 1, step_samples):
        end_idx  = start_idx + window_samples
        w_start  = pd.Timestamp(timestamps[start_idx])
        w_end    = pd.Timestamp(timestamps[end_idx - 1])
        label    = assign_label(w_start, w_end, ev_starts, ev_ends, ev_labels)                  # Assign label to the window based on event overlap
        window   = np.stack([airflow_f[start_idx:end_idx],
                             thoracic_f[start_idx:end_idx]], axis=-1)
        windows.append(window)
        labels.append(label)

    return np.array(windows, dtype=np.float32), labels                                                  # Return windows as numpy array and labels as list of strings       





def process_participant(participant_path):
    name = os.path.basename(os.path.normpath(participant_path))

    # Find required files
    airflow_path, thoracic_path, events_path = find_files(participant_path)

    # Load signals and events
    ts_air, airflow = load_signal(airflow_path)
    ts_thor, thoracic = load_signal(thoracic_path)
    ev_starts, ev_ends, ev_labels = load_events(events_path)

    # Ensure both signals have the same length
    min_len = min(len(airflow), len(thoracic))
    airflow = airflow[:min_len]
    thoracic = thoracic[:min_len]
    timestamps = ts_air[:min_len]

    # Apply bandpass filtering to remove noise
    airflow_f = bandpass_filter(airflow)
    thoracic_f = bandpass_filter(thoracic)

    # Create sliding windows and assign labels
    windows, labels = create_windows(
        airflow_f, thoracic_f, timestamps,
        ev_starts, ev_ends, ev_labels
    )

    return name, windows, labels







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", required=True)
    parser.add_argument("-out_dir", required=True)
    args = parser.parse_args()


    all_windows = []       # signal windows
    all_labels = []        # labels for each window
    all_participants = []  # participant name for each window

    
    for participant in sorted(os.listdir(args.in_dir)):                                     # Loop through each participant folder
        p_path = os.path.join(args.in_dir, participant)
        if not os.path.isdir(p_path):
            continue

        try:
  
            name, windows, labels = process_participant(p_path)
            all_windows.append(windows)
            all_labels.extend(labels)

            
            all_participants.extend([name] * len(labels))                                   # Store participant name for each window

        except Exception:
            continue

    
    X_final = np.vstack(all_windows)                                                        # Combine all participant windows into one dataset
    
    y_final = np.array(all_labels)
    p_final = np.array(all_participants)







    dataset = {
        "X":            X_final,
        "y":            y_final,
        "participants": p_final,
        "fs":           32,
        "window_sec":   30,
        "step_sec":     15,
        "channels":     ["nasal_airflow", "thoracic_movement"],
        "filter":       "butterworth_bandpass_0.17_0.4Hz_order4",
    }

# to save in pickle and csv formats

    pkl_path = os.path.join(args.out_dir, "breathing_dataset.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"\n  Saved pickle : {pkl_path}")

    df = pd.DataFrame({"participant": p_final, "label": y_final})
    csv_path = os.path.join(args.out_dir, "breathing_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV    : {csv_path}")


if __name__ == "__main__":
    main()