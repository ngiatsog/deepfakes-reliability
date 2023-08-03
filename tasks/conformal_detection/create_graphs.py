import sys
import numpy as np
import seaborn as sn
import pandas as pd

from matplotlib import pyplot as plt
from .utils import get_bayes_fns
from pathlib import Path



def plot_cm(cm, dset, dirpath):


    titles = ["real", "fake", "unsure"]

    for mode in ["nonconformal", "conformal"]:
    
        df_cm = pd.DataFrame(cm[mode], titles[:2], titles[: (cm[mode].shape[1])])
        
        plt.figure(figsize=(10,7))
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 30}, cmap="Blues") # font size

        plt.savefig(Path(dirpath) / f"reliability_{dset}_cm_{mode}.png") 
        plt.savefig(Path(dirpath) / f"reliability_{dset}_cm_{mode}.pdf") 

paths = {}
for file in Path(sys.argv[1]).glob('*_nonconformal_confusion_matrix.txt'):
    dset = file.name[:-34]
    
    nonconformal_path = str(file)
    conformal_path = nonconformal_path[:-34]+"_conformal_confusion_matrix.txt"

    cm = {
    "nonconformal": np.loadtxt(nonconformal_path)/100,
    "conformal": np.loadtxt(conformal_path)/100
    }

    pfake = np.linspace(0, 1, 50)

    colors = {
    "conformal": "k",
    "nonconformal": "k-"
    }

    # f, axes = plt.subplots(1, 3, figsize=(18, 5))
    # for mode in ["conformal", "nonconformal"]:
    #     real_inv, fake_inv = get_bayes_fns(cm[mode][:2, :2])
    #     style = colors[mode] + "-"
    #     axes[0].plot(pfake, real_inv(pfake), style, label=mode)
    #     style = colors[mode] + "-"
    #     axes[1].plot(pfake, fake_inv(pfake), style, label=mode)
    #     style = colors[mode] + "-"
    #     axes[2].plot(pfake, 0.5*real_inv(pfake) + 0.5*fake_inv(pfake), style, label=mode)
    # for ax in axes:
    #     ax.set_xlim([0, 1])
    #     ax.set_ylim([0, 1.02])
    #     ax.grid()
    #     ax.plot()
    #     ax.legend(prop={"family":"serif", "size":15})
    
    # fs = 18
    # fm = "serif"
    # # axes[0].set_xlabel("Perc of fakes", family=fm, fontsize=fs)
    # axes[0].set_ylabel("Correct real decisions", family=fm, fontsize=fs)
    # # axes[1].set_xlabel("Perc of fakes", family=fm, fontsize=fs)
    # axes[1].set_ylabel("Correct fake decisions", family=fm, fontsize=fs)
    # # axes[2].set_xlabel("Perc of fakes", family=fm, fontsize=fs)
    # axes[2].set_ylabel("Correct decisions (average)", family=fm, fontsize=fs)
    # f.supxlabel("Prior prevalence of fakes", family=fm, fontsize=fs)
    # # if args.dset == "toyforgerynetimages":
    # #     cats = "all" if args.categories is None else ", ".join([str(cat) for cat in args.categories]) 
    # #     title = f"forgerynet (categories {cats})" 
    # # else:
    # #     title = args.dset
    # # f.suptitle(title, family=fm, fontsize=fs+5)
    # f.tight_layout()

    dir_p = Path(__file__).parent / "deliverable_graphs"
    # dir_p.mkdir(exist_ok=True)

    # p = str(dir_p / (dset+".png"))
    # f.savefig(p, dpi=300)

    # p = str(dir_p / (dset+".pdf"))
    # f.savefig(p, dpi=300)

    plot_cm(cm, dset, dir_p)





