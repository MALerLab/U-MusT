import os

from omegaconf import OmegaConf, DictConfig
from easydict import EasyDict as edict
import yaml

from rqvae.models.rqtransformer.configs import RQTransformerConfig


def easydict_to_dict(obj):
    if not isinstance(obj, edict):
        return obj
    else:
        return {k: easydict_to_dict(v) for k, v in obj.items()}


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = easydict_to_dict(config)
        config = OmegaConf.create(config)
    return config


def is_stage1_arch(arch_type):
    return not ('transformer' in arch_type)


def augment_arch_defaults(arch_config):

    if arch_config.type == 'rq-vae':
        arch_defaults = OmegaConf.create(
            {
                'ema': None,
                'hparams': {
                    'loss_type': 'l1',
                    'restart_unused_codes': False,
                    'use_padding_idx': False,
                    'masked_dropout': 0.0,
                },
                'checkpointing': False,
            }
        )
    elif arch_config.type == 'rq-transformer':
        arch_defaults = RQTransformerConfig.create(arch_config)
    else:
        raise NotImplementedError

    return OmegaConf.merge(arch_defaults, arch_config)


def augment_optimizer_defaults(optim_config):

    defaults = OmegaConf.create(
        {
            'type': 'adamW',
            'max_gn': None,
            'warmup': {
                'mode': 'linear',
                'start_from_zero': (True if optim_config.warmup.epoch > 0 else False),
            },
        }
    )
    return OmegaConf.merge(defaults, optim_config)


def augment_defaults(config):

    defaults = OmegaConf.create(
        {
            'arch': augment_arch_defaults(config.arch),
            'dataset': {
                'transform': {'type': None},
            },
            'optimizer': augment_optimizer_defaults(config.optimizer),
            'experiment': {
                'test_freq': 10,
                'amp': False,
            },
        }
    )

    if 'gan' in config:
        gan_defaults = OmegaConf.merge(defaults.optimizer, config.gan.disc.get('optimizer', {}))
        defaults.gan = OmegaConf.create(
            {
                'disc': {'optimizer': gan_defaults},
            }
        )

    if not is_stage1_arch(config.arch.type):

        model_aux_path = config.vqvae.ckpt
        model_aux_config_path = os.path.join(os.path.dirname(model_aux_path), 'config.yaml')
        stage1_arch_config = load_config(model_aux_config_path).arch

        config.vqvae = stage1_arch_config
        config.vqvae.ckpt = model_aux_path

        defaults.vqvae = augment_arch_defaults(config.vqvae)
        defaults.arch.vocab_size = config.dataset.vocab_size
        defaults.experiment.sample = {'top_k': None, 'top_p': None}

        if config.get('loss', {}).get('type', '') == 'soft_target_cross_entropy':
            defaults.loss = {'temp': 1.0, 'stochastic_codes': False}
        else:
            defaults.loss = {'type': 'cross_entropy', 'temp': 1.0, 'stochastic_codes': False}

    config = OmegaConf.merge(defaults, config)

    return config


def augment_dist_defaults(config, distenv):
    config = config.copy()

    if isinstance(config.experiment.batch_size, DictConfig):
        # Handle dynamic batch sizes for different buckets
        local_batch_sizes = config.experiment.batch_size
        world_batch_sizes = {k: distenv.world_size * v for k, v in local_batch_sizes.items()}
        total_batch_sizes = config.experiment.get('total_batch_size', world_batch_sizes)

        if isinstance(total_batch_sizes, dict):
            # Verify each bucket's total batch size is divisible by world batch size
            for bucket in world_batch_sizes:
                if total_batch_sizes[bucket] % world_batch_sizes[bucket] != 0:
                    raise ValueError(f'total batch size for bucket {bucket} must be divisible by world batch size')
            grad_accm_steps = {k: total_batch_sizes[k] // world_batch_sizes[k] for k in world_batch_sizes}
        else:
            # Single total batch size value provided
            grad_accm_steps = {}
            for bucket, world_bs in world_batch_sizes.items():
                if total_batch_sizes % world_bs != 0:
                    raise ValueError(f'total batch size must be divisible by world batch size for bucket {bucket}')
                grad_accm_steps[bucket] = total_batch_sizes // world_bs

    else:
        # Original single batch size logic
        local_batch_size = config.experiment.batch_size
        world_batch_size = distenv.world_size * local_batch_size
        total_batch_size = config.experiment.get('total_batch_size', world_batch_size)

        if total_batch_size % world_batch_size != 0:
            raise ValueError('total batch size must be divisible by world batch size')
        grad_accm_steps = total_batch_size // world_batch_size

    config.optimizer.grad_accm_steps = grad_accm_steps
    config.experiment.total_batch_size = total_batch_sizes if isinstance(config.experiment.batch_size, DictConfig) else total_batch_size

    return config


def config_setup(args, distenv, config_path, extra_args=()):

    if args.eval:
        config = load_config(config_path)
        config = augment_defaults(config)

        if hasattr(args, 'test_batch_size'):
            config.experiment.batch_size = args.test_batch_size
        if not hasattr(config, 'seed'):
            config.seed = args.seed

    elif args.resume:
        config = load_config(config_path)
        if distenv.world_size != config.runtime.distenv.world_size:
            raise ValueError("world_size not identical to the resuming config")
        config.runtime = {'args': vars(args), 'distenv': distenv}

    else:  # training
        config_path = args.model_config
        config = load_config(config_path)

        extra_config = OmegaConf.from_dotlist(extra_args)
        config = OmegaConf.merge(config, extra_config)

        config = augment_defaults(config)
        config = augment_dist_defaults(config, distenv)

        config.seed = args.seed
        config.runtime = {'args': vars(args), 'extra_config': extra_config, 'distenv': distenv}

    return config
