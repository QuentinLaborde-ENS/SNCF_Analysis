import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
# ==========================
# "Thème" façon sns.whitegrid + font_scale=2.2
# ==========================
sns.set_theme(style='whitegrid', font_scale = 2.2)

rcParams.update({
    "figure.figsize": (6, 4),        # comme ton exemple
    "xtick.direction": "out",
    "ytick.direction": "out",
    "font.size": 18,          # ~ font_scale 2.2 si base ~8
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

DATA_DIR = "movement_sanity_check/input"        # <-- à adapter
 
files = [
    "Acc_X_psd.csv",
    "Acc_Y_psd.csv",
    "Acc_Z_psd.csv",
    "Gyr_X_psd.csv",
    "Gyr_Y_psd.csv",
    "Gyr_Z_psd.csv",
]

for fname in files:
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)

    # colonnes déjà construites dans notre pipeline
    f_hz = df["Frequency_Hz"].values
    psd  = df["Normalized_PSD"].values

    fig = plt.figure()  # figsize déjà fixé via rcParams

    # courbe PSD en bleu foncé, échelle log en x
    plt.semilogx(f_hz, psd, color="darkblue")

    plt.xlabel(r"Frequency (Hz)")
    plt.ylabel(r"Normalized PSD") 

    plt.xlim(0.1, 10.0)  # 10^{-1} à 10^{1} Hz
    # plt.ylim(0, 1.05)   # optionnel si tu veux forcer l’échelle verticale

    rcParams.update({'figure.autolayout': True})
    plt.show()

    out_path = f"movement_sanity_check/figures/{fname.split('.')[0]}.png"
    fig.savefig(out_path, dpi=150)
    plt.clf()
