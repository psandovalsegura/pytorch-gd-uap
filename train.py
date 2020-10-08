import argparse
import torch
from gduap import gd_universal_adversarial_perturbation, get_data_loader, get_fooling_rate, get_baseline_fooling_rate, get_model
from analyze import get_tf_uap_fooling_rate

def validate_arguments(args):
    models = ['vgg16', 'vgg19', 'googlenet', 'resnet50', 'resnet152']

    if not (args.model in models):
        print ('Argument Error: invalid network')
        exit(-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vgg16',
                        help='The network eg. vgg16')
    parser.add_argument('--prior_type', default='no_data',
                        help='Which kind of prior to use')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size to use for training and testing')
    parser.add_argument('--patience_interval', type=int, default=5,
                        help='The number of iterations to wait to verify convergence')
    parser.add_argument('--val_dataset_name', default='voc2012',
                        help='The dataset to be used as validation')
    parser.add_argument('--final_dataset_name', default='imagenet',
                        help='The dataset to be used for final evaluation')
    parser.add_argument('--id',
                        help='An identification number (e.g. SLURM Job ID) that will prefix saved files')
    parser.add_argument('--baseline', action='store_true',
                        help='Obtain a fooling rate for a baseline random perturbation')
    parser.add_argument('--tf_uap', default=None,
                        help='Obtain a fooling rate for a input TensorFlow UAP')
    args = parser.parse_args()
    validate_arguments(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model, device)

    if args.baseline:
        print("Obtaining baseline fooling rate...")
        baseline_fooling_rate = get_baseline_fooling_rate(model, device, disable_tqdm=True)
        print(f"Baseline fooling rate for {args.model}: {baseline_fooling_rate}")
        return

    if args.tf_uap:
        print(f"Obtaining fooling rate for TensorFlow UAP called {args.tf_uap} using {args.model}...")
        tf_fooling_rate = get_tf_uap_fooling_rate(args.tf_uap, model, device)
        print(f"Fooling rate for {args.tf_uap}: {tf_fooling_rate}")
        return

    # create a universal adversarial perturbation
    uap = gd_universal_adversarial_perturbation(model, args.model, args.prior_type, args.batch_size, device, args.val_dataset_name, args.patience_interval, args.id, disable_tqdm=True)

    # perform a final evaluation
    final_data_loader = get_data_loader(args.final_dataset_name)
    final_fooling_rate = get_fooling_rate(model, uap, final_data_loader, device)
    print(f"Final fooling rate on {args.final_dataset_name}: {final_fooling_rate}")

if __name__ == '__main__':
    main()
