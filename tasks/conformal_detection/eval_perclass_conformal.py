import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from inputs import *
from pytorch_lightning.callbacks import TQDMProgressBar
from pathlib import Path

from .models import *
from ..common import get_model_target_size



if __name__ == "__main__":

    parser = ArgumentParser()
    # model params
    parser.add_argument("--model-path", type=str)
    # calibration params
    parser.add_argument("--calib-alpha", type=float, default=0.1)
    parser.add_argument("--calib-qhat", type=float, default=None)
    parser.add_argument("--calib-perclass", action="store_true")
    parser.add_argument("--calib-path", type=str)
    # dset params
    parser.add_argument("--dset", type=str, default="toyforgerynetimages")
    parser.add_argument("--mode", type=str, default="test") # use whichever split for testing purposes
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--augmentation", nargs="+", default=None)
    parser.add_argument("--preprocessing", nargs="+", default=None)
    # test params
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--with-logging", action="store_true")
    args = parser.parse_args()

    task = "detection"

    attributor = Attributor.load_from_checkpoint(args.model_path, strict=False).eval()
    conformal_detector = ConformalDetector(attributor).eval()

    target_size = get_model_target_size(conformal_detector.model_type)
    
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
    if args.calibration_qhat: #0.1367940902709961
        conformal_detector.qhat = args.calibration_qhat
        conformal_detector.alpha = args.calibration_alpha
    else:
        conformal_detector.calibrate(calib_dl, args.calibration_alpha)
    print("finished calibrating!")
    print(f"\tcalibration alpha = {conformal_detector.alpha}")
    print(f"\tcalibration qhat  = {conformal_detector.qhat}\n")
    
    # probs = torch.rand(attributor.num_classes)
    # probs /= probs.sum()
    # res = conformal_detector.predict_sets(probs)
    # print(probs)
    # print(res)
    # print(res[1].sum())


    # print("testing forward")
    # x, y = next(iter(dl))
    # device = "cuda"
    # model = model.to(device).eval()
    # out = model(x.to(device))
    # print(f"y   = {y.cpu().tolist()}")
    # print(f"out = {out.cpu().tolist()}")


    logger = None
    if args.with_logging:
        log_path = str(Path(args.model_path).parent.parent.parent.parent)
        logger = pl.loggers.TensorBoardLogger(save_dir=log_path)

    trainer = pl.Trainer(accelerator="auto",
                         logger=logger,
                         callbacks=[TQDMProgressBar(refresh_rate=5)])

    torch.set_float32_matmul_precision('medium')
    
    res = trainer.test(conformal_detector, dl)
    print("**res**")  

    print(trainer.callback_metrics)