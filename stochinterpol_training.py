import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from models.utils_files.dataloading_utils import MyDataModule, log_wandb_config, MyDistributedIterableDataset, EMACallback
import argparse
from models.sdxl_vae import SDXLAELightning
from models.forward_diffusion import linear_beta_schedule, cosine_beta_schedule, get_alph_bet
# from Python.DDPM.models.denoiser_models.unet_model import Unet, DenoisingDiffusionModel
from models.denoiser_models.standard_unet import Unet, Unet_stochinterpolant_1
from models.stochinterpolmodel import StochasticInterpolentModel
import os
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.profilers import PyTorchProfiler

import wandb
from collections import OrderedDict
from functools import partial

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from types import SimpleNamespace
from datetime import date
from peft import LoraConfig


if __name__ == '__main__':

    default_config = SimpleNamespace(
        project_name='DDPM',
        model_name='Unet',
        run_name=None,
        concat_mode='concat',
        map_type='population',
        save_model=False,
        normalize=False,
        latent_diffusion=False,
        lr_min=1e-5,
        lr_max=1e-4,
        epochs=50,
        gradient_accumulation_steps=1,
        criterion='mse',
        data_type='float32',
        prediction_step=1,
        trainer_strategy='ddp',
        lr_scheduler='plateau',  # 'constant', 'cosine', 'cosine_restart', 'plateau'
        mixed_precision=False,
        gpus=1,
        cpus=1,
        save_model_vae=False,
        num_channel=3,
        batch_size=8,
        train_ratio=0.8,
        validation_ratio=0.15,
        nb_of_simulation_folders_train=80,
        nb_of_simulation_folders_valid=100,
    )

    parser = argparse.ArgumentParser()
    # parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--project_name', 
        type=str, 
        default=default_config.project_name
        )
    parser.add_argument(
        '--model_name', 
        type=str, 
        default=default_config.project_name
        )
    parser.add_argument(
        '--run_name', 
        type=str, 
        default=default_config.run_name
        )
    parser.add_argument(
        '--run_mode', 
        type=str, 
        default='train',
        choices=['train', 'validate', 'test'],
        help="Choose between train, validate or test"
        )
    parser.add_argument(
        '--concat_mode', 
        type=str, 
        default=default_config.concat_mode, 
        choices=['repeat', 'concat']
        )
    parser.add_argument(
        '--map_type', 
        type=str, 
        default=default_config.map_type,
        choices=['population', 'k', 'costhab']
        )
    parser.add_argument(
        '--epochs',
        type=int,
        default=default_config.epochs 
        )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=default_config.gradient_accumulation_steps,
        help="Nb of steps to accumulate gradients"
        )
    parser.add_argument(
        "--prediction_step",
        type=int,
        default=default_config.prediction_step,
        help="Nb of years to predict ahead"
        )
    parser.add_argument(
        '--save_model',
        action=argparse.BooleanOptionalAction,
        default=default_config.save_model,
        help="activate saving of the model"
        )
    parser.add_argument(
        '--lr_min', 
        type=float, 
        default=default_config.lr_min
        )
    parser.add_argument(
        '--lr_max', 
        type=float, 
        default=default_config.lr_max
        )
    parser.add_argument(
        '--cpus', 
        type=int, 
        default=default_config.cpus
        )

    parser.add_argument(
        '--num_channel',
        type=int, 
        default=default_config.num_channel
        )
    parser.add_argument(
        '--criterion', 
        type=str, 
        default=default_config.criterion,
        choices=['mse', 'l1']
        )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=default_config.batch_size
        )
    parser.add_argument(
        '--normalize',
        action=argparse.BooleanOptionalAction,
        default=default_config.normalize
        )
    parser.add_argument(
        '--save_model_vae',
        action=argparse.BooleanOptionalAction,
        default=default_config.save_model_vae
        )
    parser.add_argument(
        '--latent_diffusion',
        action=argparse.BooleanOptionalAction,
        default=default_config.latent_diffusion
        )
    parser.add_argument(
        '--mixed_precision',
        action=argparse.BooleanOptionalAction,
        default=default_config.mixed_precision
        )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=default_config.train_ratio
        )
    parser.add_argument(
        '--validation_ratio',
        type=float,
        default=default_config.validation_ratio
        )
    parser.add_argument(
        '--ckpt_name',
        type=str,
        default=None,
        help="Path to the checkpoint file"
        )
    parser.add_argument(
        '--trainer_strategy',
        type=str,
        default=default_config.trainer_strategy
        )
    parser.add_argument(
        '--data_type', 
        type=str, 
        default=default_config.data_type,
        choices=['float32', 'float16', 'float64']
        )
    parser.add_argument(
        '--lr_scheduler', 
        type=str, 
        default=default_config.lr_scheduler,
        choices=['constant', 'cosine', 'cosine_restart', 'plateau'],
        )
    parser.add_argument(
        '--nb_of_simulation_folders_train', 
        type=int, 
        default=default_config.nb_of_simulation_folders_train
        )
    parser.add_argument(
        '--nb_of_simulation_folders_valid', 
        type=int, 
        default=default_config.nb_of_simulation_folders_valid
        )
    args = parser.parse_args()
    before_memory = torch.cuda.memory_allocated()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device, torch.cuda.get_device_name() if torch.cuda.is_available() else "")
    if device == 'cuda':
        if "A100" in torch.cuda.get_device_name().split('-'):
            print("Running on A100")
            torch.set_float32_matmul_precision('high')  # Enable Tensor Core acceleration
    print("A100" in torch.cuda.get_device_name() if torch.cuda.is_available() else "")
    print(torch.get_float32_matmul_precision())

    print("Running training with prediction_step =", args.prediction_step)

    today = date.today()
    formatted = today.strftime("%Y_%m_%d")
    lr = [min(args.lr_min, args.lr_max), max(args.lr_min, args.lr_max)]
    
    # Hyperparameters
    if args.criterion == 'mse':
        criterion = nn.MSELoss()
    elif args.criterion == 'l1':
        criterion = nn.SmoothL1Loss() #nn.L1Loss()
    if args.data_type == 'float32':
        data_type = torch.float32
    elif args.data_type == 'float16':
        data_type = torch.float16
    elif args.data_type == 'float64':
        data_type = torch.float64
    train_threshold = int(args.train_ratio * 20)
    validation_threshold = int((args.train_ratio + args.validation_ratio) * 20)
    if args.latent_diffusion:
        nb_channels = 4
        input_dim = 128
    else :
        nb_channels = 3
        input_dim = 1024
    model_name = args.project_name + args.model_name + '_pred_{i}'.format(i=args.prediction_step) 



    train_dataset_folder = [os.path.join(os.getenv("DATA_DIR"), "landscape_{i}".format(i=i)) for i in range(1, args.nb_of_simulation_folders_train + 1)]
    validation_data_folder = [os.path.join(os.getenv("DATA_DIR"), "landscape_{i}".format(i=i)) for i in range(args.nb_of_simulation_folders_train + 1,args.nb_of_simulation_folders_train + args.nb_of_simulation_folders_valid + 1)]
    print("Train dataset folders:", train_dataset_folder)
    print("Validation dataset folders:", validation_data_folder)
    
    if args.latent_diffusion:
        vae_model_pop_path = "gpfs/workdir/pageje/Python/DDPM/models/vae_model.pth"
        vae_model_land_path = "gpfs/workdir/pageje/Python/DDPM/models/vae_model_land.pth"
        # load the vae for encoding 
        vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", 
                cache_dir = os.getenv("CACHE_DIR") if os.getenv("CACHE_DIR") is not None else None,
                in_channels = args.num_channel,
                out_channels = args.num_channel,
                torch_dtype=data_type
                )
        if os.getenv("VAE_MODEL_POP_PATH") is not None and os.getenv("VAE_MODEL_LAND_PATH") is not None:
            vae_lora_config = LoraConfig(
                r=64,
                init_lora_weights="gaussian",
                target_modules=["to_k", 
                                "to_q", 
                                "to_v", 
                                "to_out.0",
                                # "conv1", 
                                # "conv2"
                                ],
            )
            # This is the correct usage:
            vae.add_adapter(adapter_config=vae_lora_config, adapter_name="vae_lora")
            vae_model_pop_path = os.getenv("VAE_MODEL_POP_PATH")
            print("Loading VAE model for population from:", vae_model_pop_path)
            # vae_pop_pl = SDXLAELightning.load_from_checkpoint(
            #     checkpoint_path=vae_model_pop_path,
            #     model=vae,
            # )
            # vae_pop = vae_pop_pl.model
            checkpoint = torch.load(vae_model_pop_path, weights_only=False, map_location=torch.device('cpu'))
            vae_pop_pl = SDXLAELightning(model=vae)
            vae_pop_pl.load_state_dict(checkpoint.state_dict())
            vae_model_land_path = os.getenv("VAE_MODEL_LAND_PATH")
            print("Loading VAE model for landscape from:", vae_model_land_path)
            # vae_land_pl = SDXLAELightning.load_from_checkpoint(
            #     checkpoint_path=vae_model_land_path,
            #     model=vae,
            # )
            # vae_land = vae_land_pl.model
            checkpoint = torch.load(vae_model_land_path, weights_only=False, map_location=torch.device('cpu'))
            vae_land_pl = SDXLAELightning(model=vae)
            vae_land_pl.load_state_dict(checkpoint.state_dict())
            vae_pop = vae_pop_pl.model
            vae_land = vae_land_pl.model
        else:
            vae_pop = vae
            vae_land = vae
            print("No VAE model found for either population or landscape, using default VAE")
        # checkpoint = torch.load(vae_model_path, weights_only=True)
        # vae.load_state_dict(checkpoint['model_state_dict'])

        
    list_years = range(10,51, min(5,args.prediction_step+1))
    # list_years = [30]
    if args.run_mode == 'test':
        list_years = [20,30,40]
    print("List of years:", list_years)
    list_years_before_pred_step = [i for i in list_years if i < 50 - args.prediction_step]
    print("List of years before prediction step:", list_years_before_pred_step)
    data_module = MyDataModule(
        args=args,
        train_folders=train_dataset_folder,
        validation_folders=validation_data_folder,
        train_threshold=train_threshold,
        validation_threshold=validation_threshold,
        years=list_years,
        data_type=data_type
        )
    
    # dataset = MyDistributedIterableDataset(args, dataset_folder, reps=range(0, train_threshold), years=list_years)
    # loader = DataLoader(dataset, batch_size=None, num_workers=0)

    # for cond, pred in loader:
    #     print("Batch shapes:", cond.shape, pred.shape)
    #     break

    train_dataset_length = args.nb_of_simulation_folders_train * len(list_years_before_pred_step) * len(range(0, train_threshold))
    validation_dataset_length = args.nb_of_simulation_folders_valid * len(list_years_before_pred_step) * len(range(train_threshold, validation_threshold))
    test_dataset_length = args.nb_of_simulation_folders_valid * len(list_years_before_pred_step) * len(range(validation_threshold, 20))
    
    nb_batches_per_gpu_train = train_dataset_length//args.batch_size//torch.cuda.device_count()
    nb_batches_per_gpu_validation = validation_dataset_length//args.batch_size//torch.cuda.device_count()
    nb_batches_per_gpu_test = test_dataset_length//args.batch_size//torch.cuda.device_count()

    print(f"Train dataset length: {train_dataset_length}")
    print(f"Validation dataset length: {validation_dataset_length}")
    print(f"Test dataset length: {test_dataset_length}")

    print(f"Nb batches per GPU train: {nb_batches_per_gpu_train}")
    print(f"Nb batches per GPU validation: {nb_batches_per_gpu_validation}")
    print(f"Nb batches per GPU test: {nb_batches_per_gpu_test}")

    unet = Unet_stochinterpolant_1(
            dim=input_dim, #for conditioning 
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=nb_channels,
            with_time_emb=True,
            convnext_mult=2,
            GroupNorm=True,
            film_cond_dim=1024,
            )

    if os.getenv("CKPT_DIR") is None or os.getenv("CKPT_DIR") == "":     
        model_folder = os.path.join(
            './checkpoints', 
            model_name,
            # formatted
            )
        print("No CKPT_DIR environment variable found, saving to default path:", model_folder)
    else :
        model_folder = os.path.join(
            os.getenv("CKPT_DIR"), 
        )

    if args.save_model:
        os.makedirs(
            model_folder, 
            exist_ok=True
        )
    ckpt_path = None
    if args.ckpt_name is not None:
        ckpt_path = os.path.join(
            model_folder, 
            args.ckpt_name
            )
        # checkpoint = torch.load(ckpt_path, map_location='cpu')
        # unet.load_state_dict(checkpoint)

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_folder,
        monitor='val_loss',
        filename=model_name+args.run_name+'_{epoch:02d}_{val_loss:.3f}',
        save_top_k=1,
        mode='min',  # 'min' for loss, 'max' for accuracy
        save_last=True  # Save the last epoch
        )
    ema_callback = EMACallback(decay=0.9999, save_dir=model_folder, ema_filename="best_ema.ckpt")

    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=7, 
                                   mode='min',
                                    # check_on_train_epoch_end=False,  # Important if you log only at val epoch end
                                    # strict=False  # <-- Add this!
                                   )
    wandb_logger = None
    if os.getenv("WANDB_API_KEY") is not None:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=args.project_name,  # Specify your project
            name=args.run_name,
            config={                        # Track hyperparameters and metadata
                # "learning_rate": args.lr,
                # "epochs": args.epochs,
            }
            # dir=os.path.join(
        )
        wandb_logger = WandbLogger(
            project=args.project_name,
            name=model_name,
            log_model=False,
            save_dir=os.getenv("WANDB_DIR"),
            always_display=['val_loss', 'lr', 'train_loss'],
            settings=wandb.Settings(
                start_method="thread",  # Au lieu de "fork" qui peut causer des problèmes
                _disable_stats=False,
                _disable_meta=False,
            ),
            mode="offline",
            monitor_gpus=True
            )
        # wandb.define_metric("gpu_memory", summary="max")  
        # wandb.watch_called = False  
        # wandb.watch(unet, log="all", log_freq=100)  
    profiler = PyTorchProfiler(
            dirpath=".", 
            filename="pytorch_profiler", 
            export_to_chrome=True,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./tb_log'),
            record_shapes=True,
            profile_memory=True,
        )
    trainer = Trainer(
        # For debugging purposes, you can uncomment the following lines:
        # limit_train_batches=50,         # profile only a few batches first
        # limit_val_batches=1,
        # enable_checkpointing=False,    # to reduce noise during profiling
        # profiler=profiler,
        # For training:
        limit_train_batches=nb_batches_per_gpu_train,
        limit_val_batches=nb_batches_per_gpu_validation,
        limit_test_batches=nb_batches_per_gpu_test,
        logger=wandb_logger,
        callbacks=[
                checkpoint_callback,
                ema_callback,
                #    early_stopping
                   ],
        log_every_n_steps=1,
        precision="bf16-mixed",#"32-true", #if args.mixed_precision else "16-mixed",
        strategy="ddp", #args.trainer_strategy,
        num_nodes=1,
        gradient_clip_val=5.0, 
        gradient_clip_algorithm="norm",
        accelerator="gpu", 
        devices="auto",
        accumulate_grad_batches=args.gradient_accumulation_steps,
        max_epochs=args.epochs
        )
    model = StochasticInterpolentModel(
        denoiser=unet,
        # criterion=criterion,
        trainer=trainer,
        lr=lr,
        save_vae=args.save_model_vae,
        vae_pop=vae_pop if args.latent_diffusion else None,
        vae_land=vae_land if args.latent_diffusion else None,
        scheduler=args.lr_scheduler,
        )
    if os.getenv("WEIGHTS_PATH") is not None and os.getenv("WEIGHTS_PATH") != "":
        weights_path = os.getenv("WEIGHTS_PATH")
        print("Loading model weights from:", weights_path)
        checkpoint = torch.load(weights_path, weights_only=False, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        ckpt_path = None  # Avoid loading again in trainer.fit()

    if os.getenv("WANDB_API_KEY") is not None:
        wandb_logger.watch(model, log="all", log_freq=100)
        log_wandb_config(wandb_logger, args)


    if args.run_mode == 'train':
        print("Starting training...")
        trainer.fit(
            model=model,
            datamodule=data_module,
            ckpt_path=ckpt_path
        )
    elif args.run_mode == 'validate':
        print("Starting validation...")
        result = trainer.validate(model=model, datamodule=data_module)
        print(result)
    elif args.run_mode == 'test':
        print("Starting testing...")
        result = trainer.test(model=model, datamodule=data_module)
        print(result)

