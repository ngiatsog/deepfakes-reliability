import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from inputs import *
from pytorch_lightning.callbacks import TQDMProgressBar
from pathlib import Path
from matplotlib import pyplot as plt

from .utils import get_bayes_fns
from .models import DetectorViaAttribution, ConformalDetector
from ..common import get_model_target_size

eps = 0.000001

if __name__ == "__main__":

    parser = ArgumentParser()
    # model params
    parser.add_argument("--model-path", type=str)
    # calibration params
    parser.add_argument("--calib-alpha", type=float, default=0.1)
    parser.add_argument("--calib-qhat", type=float, default=None)
    parser.add_argument("--calib-qhat-perclass", nargs="+", type=float, default=None)
    parser.add_argument("--calib-perclass", action="store_true")
    parser.add_argument("--calib-path", type=str)
    # dset params
    parser.add_argument("--dset", type=str, default="toyforgerynetimages")
    parser.add_argument("--mode", type=str, default=None) # use whichever split for testing purposes
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--augmentation", nargs="+", default=None)
    parser.add_argument("--preprocessing", nargs="+", default=None)
    # test params
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--with-logging", action="store_true")
    parser.add_argument("--res-dir", type=str)
    args = parser.parse_args()

    task = "detection"

    conformal_detector = ConformalDetector.load_from_checkpoint(args.model_path)
    nonconformal_detector = DetectorViaAttribution.load_from_checkpoint(args.model_path)
    target_size = get_model_target_size(nonconformal_detector.model_type)

    ds = get_imagelabel_dataset(args.dset,
                                mode=args.mode,
                                categories=args.categories,
                                task=task,
                                target_size=target_size,
                                augmentation=args.augmentation,
                                preprocessing=args.preprocessing)
    dl = get_dataloader(ds, args.batch_size, strategy="none")

    calib_ds = load_imagelabel_dataset(args.calib_path,
                            target_size=target_size)
    calib_dl = get_dataloader(calib_ds, args.batch_size, strategy="none")

    print("start calibrating")
    if (not args.calib_perclass) and args.calib_qhat: #0.1367940902709961
        conformal_detector.qhat = args.calib_qhat
        conformal_detector.alpha = args.calib_alpha
    elif args.calib_perclass and args.calib_qhat_perclass:
        conformal_detector.qhat = torch.tensor(args.calib_qhat_perclass)
        conformal_detector.alpha = args.calib_alpha
    else:
        conformal_detector.calibrate(calib_dl, args.calib_alpha, args.calib_perclass)
        
    print("finished calibrating!")
    print(f"\tcalibration alpha = {conformal_detector.alpha}")
    print(f"\tcalibration qhat  = {conformal_detector.qhat}\n")


    logger = None
    if args.with_logging:
        log_path = str(Path(args.model_path).parent.parent.parent.parent)
        logger = pl.loggers.TensorBoardLogger(save_dir=log_path)

    trainer = pl.Trainer(accelerator="auto",
                         logger=logger,
                         
                         callbacks=[TQDMProgressBar(refresh_rate=5)])

    torch.set_float32_matmul_precision('medium')
    
    cm = {}
    # nonconformal res
    for mode, detector in zip(["nonconformal", "conformal"], [nonconformal_detector, conformal_detector]):
        trainer.test(detector, dl)

        max_j = 2 if mode == "nonconformal" else 3
        cm[mode] = np.zeros((2, max_j))
        for i in range(2):
            for j in range(max_j):
                cm[mode][i, j] = max(trainer.callback_metrics[f"p{j}_cond{i}"], eps)


    # plot 
    pfake = np.linspace(0, 1, 50)

    colors = {
        "conformal": "r",
        "nonconformal": "k"
    }

    f, axes = plt.subplots(1, 3, figsize=(17, 5))
    for mode in ["conformal", "nonconformal"]:
        real_inv, fake_inv = get_bayes_fns(cm[mode][:2, :2])
        style = colors[mode] + "-"
        axes[0].plot(pfake, real_inv(pfake), style, label=mode)
        style = colors[mode] + "-"
        axes[1].plot(pfake, fake_inv(pfake), style, label=mode)
        style = colors[mode] + "-"
        axes[2].plot(pfake, 0.5*real_inv(pfake) + 0.5*fake_inv(pfake), style, label=mode)
    for ax in axes:
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid()
        ax.plot()
        ax.legend()

    fs = 12
    fm = "serif"
    axes[0].set_xlabel("Perc of fakes", family=fm, fontsize=fs)
    axes[0].set_ylabel("Correct real decisions (posterior)", family=fm, fontsize=fs)
    axes[1].set_xlabel("Perc of fakes", family=fm, fontsize=fs)
    axes[1].set_ylabel("Correct fake decisions (posterior)", family=fm, fontsize=fs)
    axes[2].set_xlabel("Perc of fakes", family=fm, fontsize=fs)
    axes[2].set_ylabel("Correct decisions (posterior)", family=fm, fontsize=fs)

    if args.dset == "toyforgerynetimages":
        cats = "all" if args.categories is None else ", ".join([str(cat) for cat in args.categories]) 
        title = f"forgerynet (categories {cats})" 
    else:
        title = args.dset
    f.suptitle(title, family=fm, fontsize=fs+5)
    f.tight_layout()

    dir_p = Path(__file__).parent / "conformal_vs_nonconformal_res" / args.res_dir
    dir_p.mkdir(exist_ok=True)
    
    p = str(dir_p / (title+".png"))
    f.savefig(p, dpi=300)
    
    p = str(dir_p / (title+".pdf"))
    f.savefig(p, dpi=300)
    
    p = str(dir_p / (title+"_conformal_confusion_matrix.txt"))
    np.savetxt(p , 100*cm["conformal"], fmt="%.2f")

    p = str(dir_p / (title+"_nonconformal_confusion_matrix.txt"))
    np.savetxt(p, 100*cm["nonconformal"], fmt="%.2f")
