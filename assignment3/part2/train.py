from cifar10_models import resnet18
from utils import load_cifar10, train, test
from adversarial_attack import test_attack
from globals import STANDARD, FGSM, PGD, CIFAR10_LABELS, ALPHA, EPSILON, NUM_ITER

import argparse
import torch
import matplotlib.pyplot as plt
import os

def main(args):
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    strategy_args = {STANDARD: None, 
                 FGSM: {ALPHA: args.alpha_fgsm, EPSILON: args.epsilon_fgsm}, 
                 PGD: {ALPHA: args.alpha_pgd, EPSILON: args.epsilon_pgd, NUM_ITER: args.num_iter_pgd}}
    for training_strategy in args.train_strats:
        print(f"training_strategy: {training_strategy}")
        print("Loading model")
        model = resnet18(pretrained=args.pretrained).to(device)
        print("Loading data")
        trainloader, validloader, testloader, _ = load_cifar10(batch_size=args.batch_size,
                                                               valid_ratio=args.valid_ratio,
                                                               augmentations=args.augmentations)
        if training_strategy == STANDARD and args.pretrained:
            print("Skipping training for standard pretrained model")
        else:
            print("Training model")
            model = train(model, trainloader, validloader, num_epochs=args.num_epochs, 
                        defense_strategy=training_strategy, 
                        defense_args=strategy_args[training_strategy])
        
        print("Testing model")
        test(model, testloader)
        print("Testing adversarial attacks")
        if not args.test_crossover_defense:
            if training_strategy == STANDARD: 
                for attack in [FGSM, PGD]:	
                        adv_acc, adv_examples = test_attack(model, testloader, attack, strategy_args[attack])
                        visualise(args, adv_examples, training_strategy, attack, save_dir=args.save_dir)
            else:
                adv_acc, adv_examples = test_attack(model, testloader, training_strategy, strategy_args[training_strategy])
                visualise(args, adv_examples, training_strategy, training_strategy, save_dir=args.save_dir)
        else:
            for attack in [FGSM, PGD]:	
                adv_acc, adv_examples = test_attack(model, testloader, attack, strategy_args[attack])
                visualise(args, adv_examples, training_strategy, attack, save_dir=args.save_dir)



def visualise(args, adv_examples, training_strategy, attack, save_dir = ''):
    output_dir = "adversarial_examples" + save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.visualise:
        for i, adv_ex in enumerate(adv_examples):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle(f"Model: {training_strategy}, Attack: {attack}")
            label_1 = CIFAR10_LABELS[adv_ex[0]]
            ax1.set_title(f"Original Image\nPredicted label: {label_1}")
            im1 = adv_ex[2][0].numpy().transpose(1, 2, 0)
            ax1.imshow(im1)
            ax1.axis('off')
            label_2 = CIFAR10_LABELS[adv_ex[1]]
            ax2.set_title(f"Adversasrial Image\n Predicted label: {label_2}")
            im2 = adv_ex[3][0].numpy().transpose(1, 2, 0)
            ax2.imshow(im2)
            ax2.axis('off')
            plt.savefig(output_dir + f"/train-{training_strategy}_att-{attack}_{i+1}.png")
            plt.close(fig)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test ResNet18 on CIFAR-10 dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing')
    parser.add_argument('--valid_ratio', type=float, default=0.75, help='Validation set ratio')
    parser.add_argument('--augmentations', action='store_true', help='Use data augmentations for training')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs for training')
    parser.add_argument('--train_strats', nargs='+', choices=[STANDARD, FGSM, PGD], default=[STANDARD], help='List of loss types to use for training')
    parser.add_argument('--visualise', action='store_true', help='Visualise adversarial examples')
    parser.add_argument('--epsilon_fgsm', type=float, default=0.1, help='Epsilon for FGSM attack')
    parser.add_argument('--alpha_fgsm', type=float, default=0.5, help='Alpha for FGSM attack')
    parser.add_argument('--epsilon_pgd', type=float, default=0.01, help='Epsilon for PGD attack')
    parser.add_argument('--alpha_pgd', type=float, default=2, help='Alpha for PGD attack')
    parser.add_argument('--num_iter_pgd', type=int, default=10, help='Number of iterations for PGD attack')
    parser.add_argument('--save_dir', type=str, default='', help='Directory to save'	)
    parser.add_argument('--test_crossover_defense', action='store_true', help='Test crossover defense')
    args = parser.parse_args()
    main(args)