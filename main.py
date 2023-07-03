
import argparse
import time
import torch
import numpy as np
import torch.optim as optim
from skimage.io import imread, imshow
# custom modules

from utils import get_model, to_device, prepare_dataloader, transform2Dto3D
from loss import TwinLoss
import os
import torch.nn.functional as F
import PIL.Image as pil
from torchvision import transforms
import cv2
import matplotlib.cm as cm
from sklearn.metrics import r2_score

# plot params

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15, 10)

file_dir = os.path.dirname(__file__)  # the directory that main.py resides in


def return_arguments():
    parser = argparse.ArgumentParser(description='PyTorch ')

    parser.add_argument('--data_dir',
                        type=str,
                        help='path to the dataset folder',
                        default='/disk_three/coffbea-2023/coffbea-laser/data/')
    parser.add_argument('--model_path',
                        default=os.path.join(file_dir, "weights"),
                        help='path to the trained model')
    parser.add_argument('--is_stereo', default=True,
                        help='input stereo images')
    parser.add_argument('--input_height', type=int, help='input height',
                        default=306)
    parser.add_argument('--input_width', type=int, help='input width',
                        default=230)
    parser.add_argument('--full_height', type=int, help='input height',
                        default=920)
    parser.add_argument('--full_width', type=int, help='input width',
                        default=1224)
    parser.add_argument('--size', type=tuple, help='input size',
                        default=(896, 896))
    parser.add_argument('--model', default='stereo_vit_lstm',
                        help='mono_res, mono_vit, mono_vit_mlp, mono_res_mlp, '
                             'stereo_res, stereo_vit, stereo_vit_mlp, stereo_res_mlp, '
                             'stereo_res_lstm, stereo_vit_lstm, ')
    parser.add_argument('--resume', default=None,
                        help='load weights to continue train from where it last stopped')
    parser.add_argument('--load_weights_folder', default=os.path.join(file_dir, "weights"),
                        help='folder to load weights to continue train from where it last stopped')
    parser.add_argument('--pretrained', default=True,
                        help='Use weights of pretrained model')
    parser.add_argument('--mode_flag', default='train',
                        help='mode: train or test (default: train)')
    parser.add_argument('--epochs', type=int, default=800,
                        help='number of total epochs to run')
    parser.add_argument('--startEpoch', type=int, default=0,
                        help='number of total epochs to run')
    parser.add_argument('--testepoch', type=str, default='border_cpt',
                        help='number of total epochs to test')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='initial learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=66,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--adjust_lr', default=True,
                        help='apply learning rate decay or not\
                        (default: True)')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='choose cpu or cuda:0 device"')
    parser.add_argument('--do_augmentation', default=True,
                        help='do augmentation of images or not')
    parser.add_argument('--augment_parameters', default=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                        help='lowest and highest values for gamma,\
                        brightness and color respectively')
    parser.add_argument('--print_weights', default=False,
                        help='print weights of every layer')
    parser.add_argument('--input_channels', default=3,
                        help='Number of channels in input tensor')
    parser.add_argument('--num_workers', default=4,
                        help='Number of workers in dataloader')
    parser.add_argument('--use_multiple_gpu', default=True)

    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, learning_rate):
    if epoch >= 300 and epoch < 400:
        lr = learning_rate / 2
    elif epoch >= 400:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Model:
    def __init__(self, args):
        self.args = args
        # create weight folder
        if os.path.isdir(args.model_path):
            print('Weights folder exists')
        else:
            print('Weights folder create')
            os.makedirs(args.model_path)

        # Set up model
        self.device = args.device
        self.model = get_model(args.model, pretrained=args.pretrained)
        self.model = self.model.to(self.device)
        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)

        # resume
        self.best_val_loss = float('Inf')

        if args.mode_flag == 'train':
            self.loss_function = TwinLoss(
                L1_w=0.5,
                L2_w=0.5).to(self.device)

            self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

            if args.resume is not None:
                self.load_model_continue_train(os.path.join(self.args.model_path, 'weights_last.pt'))
                self.args.startEpoch = self.startEpoch
                self.best_val_loss = self.best_val_loss

            # load val data first
            self.val_n_img, self.val_loader = prepare_dataloader(data_directory=args.data_dir,
                                                                 is_stereo=args.is_stereo,
                                                                 batch_size=args.batch_size,
                                                                 num_workers=args.num_workers,
                                                                 size=args.size,
                                                                 mode_flag='val')
            # Load train data
            self.train_n_img, self.train_loader = prepare_dataloader(data_directory=args.data_dir,
                                                                     is_stereo=args.is_stereo,
                                                                     batch_size=args.batch_size,
                                                                     num_workers=args.num_workers,
                                                                     size=args.size,
                                                                     mode_flag='train')
        else:
            args.test_model_path = os.path.join(self.args.model_path, args.testepoch + '.pth')
            self.model.load_state_dict(torch.load(args.test_model_path))
            args.batch_size = 1
            self.test_n_img, self.test_loader = prepare_dataloader(data_directory=args.data_dir,
                                                                   is_stereo=args.is_stereo,
                                                                   batch_size=args.batch_size,
                                                                   num_workers=args.num_workers,
                                                                   size=args.size,
                                                                   mode_flag='test')

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def train(self):
        val_losses = []
        running_val_loss = 0.0
        self.model.eval()
        for data in self.val_loader:
            data = to_device(data, self.device)  # dict
            images = data['images']
            points = torch.flatten(data['points'], 1)
            left_gt = data['labels']

            left_pred = self.model(images, points)
            loss = self.loss_function(left_pred, left_gt)

            val_losses.append(loss.item())
            running_val_loss += loss.item()

        running_val_loss /= self.val_n_img / self.args.batch_size  # todo: check again
        print('Val_loss:', running_val_loss)
#
        for epoch in range(self.args.startEpoch, self.args.epochs):
            if self.args.adjust_lr:
                adjust_learning_rate(self.optimizer, epoch,
                                     self.args.learning_rate)
            c_time = time.time()
            running_loss = 0.0
            training_losses = []
            self.model.train()
            for data in self.train_loader:
                # Load data
                data = to_device(data, self.device)
                images = data['images']
                points = torch.flatten(data['points'], 1)
                left_gt = data['labels']

                # One optimization iteration
                self.optimizer.zero_grad()
                left_pred = self.model(images, points)
                loss = self.loss_function(left_pred, left_gt)
                loss.backward()
                self.optimizer.step()
                training_losses.append(loss.item())
                running_loss += loss.item()

            running_val_loss = 0.0
            self.model.eval()
            for data in self.val_loader:
                data = to_device(data, self.device)
                images = data['images']
                points = torch.flatten(data['points'], 1)
                left_gt = data['labels']

                left_pred = self.model(images, points)
                loss = self.loss_function(left_pred, left_gt)

                val_losses.append(loss.item())
                running_val_loss += loss.item()

            # Estimate loss per image
            running_loss /= self.train_n_img / self.args.batch_size
            running_val_loss /= self.val_n_img / self.args.batch_size
            print(
                'Epoch:',
                epoch + 1,
                'train_loss:',
                running_loss,
                'val_loss:',
                running_val_loss,
                'time:',
                round(time.time() - c_time, 3),
                's',
            )

            # save weights for every 5 epoch
            if epoch % 5 == 0:
                self.save(os.path.join(self.args.model_path, 'epoch{}.pth'.format(str(epoch))))
            # if running_val_loss < best_val_loss:
            if running_val_loss < self.best_val_loss:
                self.save(os.path.join(self.args.model_path, 'border_cpt.pth'))
                self.best_val_loss = running_val_loss
                self.save_continue_train(epoch, running_val_loss, 'weights_cpt.pt')
                print('Model_saved')

            self.save(os.path.join(self.args.model_path, 'border_last.pth'))
            self.save_continue_train(epoch, running_loss, 'weights_last.pt')

        print('Finished Training.')
        # self.save(os.path.join(self.args.model_path, 'train_end.pth'))
        self.save_continue_train(self.args.epochs, running_val_loss, 'train_end.pt')

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def save_continue_train(self, epoch, loss, path):
        save_path = os.path.join(self.args.model_path, path)
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'best_val_loss': self.best_val_loss
                    }, save_path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def load_model_continue_train(self, path):
        assert os.path.isfile(path), \
            "Cannot find folder {}".format(path)
        print("loading model from folder {}".format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.startEpoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

    def test(self):
        self.model.eval()
        error2d = []
        r2pred = []
        r2gt = []
        for data in self.test_loader:
            data = to_device(data, self.device)
            images = data['images']
            points = torch.flatten(data['points'], 1)
            left_gt = data['labels']

            left_pred = self.model(images, points).cpu().detach().numpy()
            left_gt = left_gt.cpu().detach().numpy()

            # loss 2d
            loss_2d = np.linalg.norm(left_pred - left_gt)
            error2d.append(loss_2d)

            # R2
            r2pred.append(left_pred[0])
            r2gt.append(left_gt[0])

        # error on 2d
        mean = np.mean(error2d)
        std = np.std(error2d)
        median = np.median(error2d)
        print("\n  " + ("{:>8} | " * 3).format("2d_mean", "2d_std", "2d_median"))
        print(("&{: 8.1f}  " * 3).format(mean, std, median) + "\\\\")
        print("\n-> Done!")

        # R2
        r_square = r2_score(r2pred, r2gt)
        print('r_square = ', r_square)


def main():
    args = return_arguments()
    if args.mode_flag == 'train':
        model = Model(args)
        model.train()
    elif args.mode_flag == 'test':
        model_test = Model(args)
        model_test.test()


if __name__ == '__main__':
    main()




