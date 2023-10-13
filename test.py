import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import itertools

from Generator import Generator
from Discriminator import Discriminator

class Evaluation():
    def __init__(self):
        self.batch_size = 16
        self.epoch = 2000
        self.loader_list = []
        self.loss_value = []
        self.negative_score = []
        self.positive_score = []
    
    def make_datasets(self):

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        test_dataset_negative = datasets.MNIST(root = "../", train = False, download = True, transform = transform)
        test_dataset_positive = datasets.MNIST(root = "../", train = False, download = True, transform = transform)

        # 正常品(negative)のデータセット
        negative_test_mask = (test_dataset_negative.targets == 7)
        test_dataset_negative.data = test_dataset_negative.data[negative_test_mask]
        test_dataset_negative.targets = test_dataset_negative.targets[negative_test_mask]
        self.negative_num = len(test_dataset_negative.targets)
        test_loader_negative = DataLoader(test_dataset_negative, batch_size = self.batch_size, shuffle = False)

        # 異常品(positive）のデータセット
        positive_test_mask = (test_dataset_positive.targets != 7)
        test_dataset_positive.data = test_dataset_positive.data[positive_test_mask]
        test_dataset_positive.targets = test_dataset_positive.targets[positive_test_mask]
        test_dataset_positive.data = test_dataset_positive.data[:self.negative_num]
        test_dataset_positive.targets = test_dataset_positive.targets[:self.negative_num]
        self.positive_num = len(test_dataset_positive.targets)
        test_loader_positive = DataLoader(test_dataset_positive, batch_size = self.batch_size, shuffle = False)

        self.loader_list.append(test_loader_negative)
        self.loader_list.append(test_loader_positive)
        
    def anomaly_score(self, input_img, output_img, D):

        Lambda = 0.1

        residual_loss = torch.abs(input_img - output_img)
        residual_loss = residual_loss.view(residual_loss.size()[0], -1)
        residual_loss = torch.sum(residual_loss, dim = 1)

        _, input_feature = D(input_img)
        _, output_feature = D(output_img)

        feature_loss = torch.abs(input_feature - output_feature)
        feature_loss = torch.sum(feature_loss, dim = 1)

        loss = (1 - Lambda) * residual_loss + Lambda * feature_loss

        total_loss = torch.sum(loss)

        return total_loss, loss
    
    def save_fig(self, img, generate_img, state):

        for i in range(0, 8):
            plt.subplot(4, 4, i + 1)
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            plt.imshow(img[i][0].cpu().detach().numpy(), cmap = "gray")
            plt.subplot(4, 4, 8 + i + 1)
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            plt.imshow(generate_img[i][0].cpu().detach().numpy(), cmap = "gray")
            plt.savefig("./eval_generate_img/AnoGAN_mnist_eval_" + state + ".png")
        
        plt.clf()

    def eval_model(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.make_datasets()

        update_G = Generator(noise_dim = 32).to(device)
        update_G.load_state_dict(torch.load("AnoGAN_mnist_generator.pth", map_location = device))

        update_D = Discriminator().to(device)
        update_D.load_state_dict(torch.load("AnoGAN_mnist_discriminator.pth", map_location = device))        

        for index1, loader in enumerate(self.loader_list):
            for index2, (imges, _) in enumerate(tqdm(loader)):
                
                batch_size = imges.size()[0]

                imges = imges.to(device)

                z = torch.randn(batch_size, 32).to(device)
                z.requires_grad = True
                z_optimizer = torch.optim.Adam([z], lr = 0.001)

                for _ in range(self.epoch):
                    fake_img = update_G(z)
                    loss_sum, _ = self.anomaly_score(imges, fake_img, update_D)

                    z_optimizer.zero_grad()
                    loss_sum.backward()
                    z_optimizer.step()

                update_G.eval()
                generate_img = update_G(z)

                if index1 == 0:
                    _, loss = self.anomaly_score(imges, generate_img, update_D)
                    self.negative_score.append(loss.cpu().detach().numpy())

                    if index2 == 0:
                        self.save_fig(imges, generate_img, "negative")
                
                else:
                    _, loss = self.anomaly_score(imges, generate_img, update_D)
                    self.positive_score.append(loss.cpu().detach().numpy())

                    if index2 == 0:
                        self.save_fig(imges, generate_img, "positive")
    
if __name__ == '__main__':

    path = "/src/AnoGAN/mnist"
    os.chdir(path)

    evaluation = Evaluation()
    evaluation.eval_model()
    negative = evaluation.negative_score
    positive = evaluation.positive_score

    negative = list(itertools.chain.from_iterable(negative))
    positive = list(itertools.chain.from_iterable(positive))
    print("positive_min : {} || positive_max : {}".format(min(positive), max(positive)))
    print("negative_min : {} || negative_max : {}".format(min(negative), max(negative)))

    plt.hist(negative, bins = 40, range = (0, 350), alpha = 0.5, ec = "black", label = "normal", color = "deepskyblue")
    plt.hist(positive, bins = 40, range = (0, 350), alpha = 0.5, ec = "black", label = "anomaly", color = "tomato")
    # plt.vlines(evaluation.threshold, ymin = 0, ymax = 150, color = "limegreen", linestyles = "dashdot", label = "threshold")
    plt.xlabel("anomaly score")
    plt.legend()

    plt.savefig("./distribution.png")