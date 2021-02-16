import argparse
from torch.utils.data import DataLoader

from data_utils import create_examples
from trainer import Trainer
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(args):
    print(args)

    # Build DataLoader
    train_dataset, test_dataset = create_examples(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    # Build Trainer
    trainer = Trainer(args, train_loader, test_loader)
    # Train & Validate
    for epoch in range(1, args.epochs+1):
        trainer.train(epoch)
        trainer.validate(epoch)
        trainer.save(epoch, args.output_model_prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir',  type=str, 
                        default='/home/PublicDir/temp_user/eeg_test/dataset/SEED/ExtractedFeatures/de_LDS',
                        help='path to the folder that contains feature.npy, label.npy, cumulative.npy')
    parser.add_argument('--train_percentage',  type=int, 
                        default=0.6,
                        help='portion of training samples for each subject')
    parser.add_argument('--normalization',     type=bool,  
                        default=True, 
                        help='do normalization for train/test dataset by the rule learned from train dataset')
    parser.add_argument('--sample_per_input', type=int,
                        default=5, 
                        help='number of samples for each sample, or length of the sentence')
    parser.add_argument('--sample_len', type=int,
                        default=310, 
                        help='length of each sample')
    parser.add_argument('--corrupt_probability', type=float,
                        default=0.05,
                        help='the probability that a cell is corrupted in the generation task')
    parser.add_argument('--generation_weight', type=float,
                        default=0.1,
                        help='a constant that determines the importance of generation task in the multi-task training. total cost = generation_cost * generation_weight + classification_cost * (1 - generation_weight)')
    parser.add_argument('--output_model_prefix', type=str,
                        default='model_transformer',  
                        help='output model name prefix')
    # Input parameters
    parser.add_argument('--batch_size',     default=32,   type=int,   help='batch size')
    # Train parameters
    parser.add_argument('--epochs',         default=50,   type=int,   help='the number of epochs')
    parser.add_argument('--lr',             default=1e-6, type=float, help='learning rate')
    parser.add_argument('--no_cuda',        action='store_true')
    # Model parameters
    parser.add_argument('--n_layers',       default=3,    type=int,   help='the number of heads in the multi-head attention network')
    parser.add_argument('--n_attn_heads',   default=8,    type=int,   help='the number of multi-head attention heads')
    parser.add_argument('--dropout',        default=0.1,  type=float, help='the residual dropout value')
    parser.add_argument('--ffn_hidden',     default=512, type=int,   help='the dimension of the feedforward network')
    args = parser.parse_args()
    
    main(args)