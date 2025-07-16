from torch.utils.data import DataLoader


def get_dataloader(args):
    return get_dataloader_from_dataconfig(args.data)


def get_dataloader_from_dataconfig(args_data):

    if "cfm" in args_data.name:
        from datasets_wds.web_dataloader_cfm import SimpleImageDataset
    else:
        from datasets_wds.web_dataloader_v2 import SimpleImageDataset
    datamod = SimpleImageDataset(**args_data)
    if args_data.subset == "train":
        loader = datamod.train_dataloader()
    elif args_data.subset == "val":
        loader = datamod.eval_dataloader()
    else:
        raise ValueError(f"subset {args_data.subset} not supported")
    return loader
