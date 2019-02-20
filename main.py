import argparse
import experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # NN
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--trained', type=str, default='')
    parser.add_argument('--slope', type=float, default=0.2)
    # train
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=256)
    # misc
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='outputs/garbage')
    parser.add_argument('--message', '-m',  type=str, default='')
    args, unknown = parser.parse_known_args()
    experiment.run(args)
