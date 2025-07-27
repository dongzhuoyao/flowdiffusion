from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(args_data):
    # MNIST dataset loader using PyTorch
        transform = transforms.Compose([
            transforms.Resize((args_data.resolution, args_data.resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # MNIST is grayscale
        ])
        
        if args_data.subset == "train":
            dataset = datasets.MNIST(
                root=args_data.data_dir if hasattr(args_data, 'data_dir') else './data',
                train=True,
                download=True,
                transform=transform
            )
        elif args_data.subset == "val":
            dataset = datasets.MNIST(
                root=args_data.data_dir if hasattr(args_data, 'data_dir') else './data',
                train=False,
                download=True,
                transform=transform
            )
        else:
            raise ValueError(f"subset {args_data.subset} not supported")
        
        loader = DataLoader(
            dataset,
            batch_size=args_data.batch_size if hasattr(args_data, 'batch_size') else 32,
            shuffle=args_data.subset == "train",
            num_workers=args_data.num_workers if hasattr(args_data, 'num_workers') else 4,
            pin_memory=True
        )
        return loader


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
