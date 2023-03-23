import argparse
from config import RANDOM_SEED
from dataset import Dataset
import numpy as np
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from model import build_densenet121_model, build_efficientnet_model
from optimizer import build_sgd_optimizer, build_adam_optimizer
from utils import str2bool

dataset = None


def run_experiment(config=None, log_to_wandb=True, verbose=0):
    global dataset

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # check if config was provided
    if config is None:
        raise Exception("Not config provided.")
    print("[INFO] Configuration:", config, "\n")

    # check if dataset was provided
    if dataset is None:
        raise Exception("Dataset not provided.")

    # generate train dataset
    deterministic = config['augmentation']
    train_dataset = dataset.get_training_set(
        batch_size=config['batch_size'],
        buffer_size=dataset.num_train_examples,
        repeat=False,
        deterministic=deterministic,
        augmentation=config['augmentation'],
        pipeline=config['pipeline'])

    # generate val dataset
    validation_dataset = dataset.get_validation_set(
        batch_size=config['batch_size'],
        pipeline=config['pipeline'])

    # describe dataset distribution
    print("[INFO] Dataset Total examples:", dataset.num_total_examples)
    print("[INFO] Dataset Training examples:", dataset.num_train_examples)
    print("[INFO] Dataset Validation examples:", dataset.num_val_examples)

    # describe input shape
    input_shape = [None, dataset.input_width, 2]
    print("[INFO] Input Shape:", input_shape)

    # setup optimizer
    if config["optimizer"] == "sgd":
        optimizer = build_sgd_optimizer(initial_learning_rate=config['initial_learning_rate'],
                                        maximal_learning_rate=config['maximal_learning_rate'],
                                        momentum=config['momentum'],
                                        nesterov=config['nesterov'],
                                        step_size=config['step_size'],
                                        weight_decay=config['weight_decay'])
    elif config["optimizer"] == "adam":
        optimizer = build_adam_optimizer(initial_learning_rate=config['initial_learning_rate'],
                                         maximal_learning_rate=config['maximal_learning_rate'],
                                         step_size=config['step_size'],
                                         weight_decay=config['weight_decay'],
                                         epsilon=config['epsilon'])

    # setup model
    if config['backbone'] == "densenet":
        model = build_densenet121_model(input_shape=input_shape,
                                        dropout=config['dropout'],
                                        optimizer=optimizer,
                                        pretraining=config['pretraining'],
                                        use_loss=config['use_loss'],
                                        growth_rate=config['growth_rate'],
                                        attention=config['use_attention'],
                                        densenet_depth=config['densenet_depth'])
    elif config['backbone'] == "efficientnet":
        model = build_efficientnet_model(input_shape=input_shape,
                                         dropout=config['dropout'],
                                         optimizer=optimizer,
                                         pretraining=config['pretraining'],
                                         use_loss=config['use_loss'])
    else:
        raise Exception("Model unknown")

    # Print summary of the model
    model.summary()

    # setup callbacks
    callbacks = []
    if log_to_wandb:
        wandb_callback = WandbCallback(
            monitor="val_top_1",
            mode="max",
            save_model=False
        )
        callbacks.append(wandb_callback)

    # train model
    model.fit(train_dataset,
              epochs=config['num_epochs'],
              verbose=verbose,
              validation_data=validation_dataset,
              callbacks=callbacks)

    # get the logs of the model
    return model.history


def agent_fn(config, project, entity, verbose=0):
    wandb.init(entity=entity, project=project, config=config,
               reinit=True, settings=wandb.Settings(code_dir="."))
    _ = run_experiment(config=wandb.config, log_to_wandb=True, verbose=verbose)
    wandb.finish()


def main(args):
    global dataset

    dataset = Dataset()
    steps_per_epoch = np.ceil(dataset.num_train_examples / args.batch_size)
    config = {
        'backbone': args.backbone,
        'pretraining': args.pretraining,
        'dropout': args.dropout,
        'growth_rate': args.growth_rate,
        'use_attention': args.use_attention,
        'use_loss': args.use_loss,
        'densenet_depth': args.densenet_depth,

        'optimizer': args.optimizer,
        'initial_learning_rate': args.lr_min,
        'maximal_learning_rate': args.lr_max,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': args.weight_decay,
        'step_size': int(args.num_epochs / 2) * steps_per_epoch,
        'epsilon': args.epsilon,

        'augmentation': args.augmentation,
        'batch_size': args.batch_size,
        'pipeline': args.pipeline,
        'num_epochs': args.num_epochs,
    }

    agent_fn(config=config, project=args.project,
             entity=args.entity, verbose=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training and validation.')
    parser.add_argument('--entity', type=str,
                        help='Entity', default='davidlainesv')
    parser.add_argument('--project', type=str,
                        help='Project name', default='autsl-validation')

    parser.add_argument('--backbone', type=str,
                        help='Backbone method: \'densenet\', \'mobilenet\'',
                        default='densenet')
    parser.add_argument('--pretraining', type=str2bool,
                        help='Add pretraining', default=False)
    parser.add_argument('--dropout', type=float,
                        help='Dropout at the final layer', default=0)
    parser.add_argument('--growth_rate', type=int,
                        help='Growth rate of the DenseNet-121', default=12)
    parser.add_argument('--use_attention', type=str,
                        help='Attention module: \'se\', \'cbam\'',
                        default=None)
    parser.add_argument('--use_loss', type=str,
                        help='Loss function', default="crossentropy")
    parser.add_argument('--densenet_depth', type=int,
                        help='DenseNet depth', default=121)

    parser.add_argument('--optimizer', type=str,
                        help='Optimizer: \'sgd\', \'adam\'', default='sgd')
    parser.add_argument('--lr_min', type=float,
                        help='Minimum learning rate', default=0.001)
    parser.add_argument('--lr_max', type=float,
                        help='Minimum learning rate', default=0.01)
    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay', default=0)
    parser.add_argument('--epsilon', type=float,
                        help='Epsilon (only for Adam optimization)', default=1e-7)

    parser.add_argument('--augmentation', type=str2bool,
                        help='Add augmentation', default=False)
    parser.add_argument('--batch_size', type=int,
                        help='Batch size of training and testing', default=64)
    parser.add_argument('--pipeline', type=str,
                        help='Pipeline', default="default")
    parser.add_argument('--num_epochs', type=int,
                        help='Number of epochs', default=24)

    args = parser.parse_args()

    print(args)

    main(args)
