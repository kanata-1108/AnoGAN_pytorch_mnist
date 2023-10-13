import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

from Generator import Generator
from Discriminator import Discriminator

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root = "../", train = True, download = True, transform = transform)
train_mask = (train_dataset.targets == 7)
train_dataset.data = train_dataset.data[train_mask]
train_dataset.targets = train_dataset.targets[train_mask]
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

def weights_init(m):

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train(dataloader, epochs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    z_dim = 32

    G = Generator(noise_dim = z_dim)
    D = Discriminator()

    G.apply(weights_init)
    D.apply(weights_init)

    G.to(device)
    D.to(device)

    G_lr, D_lr = 0.0002, 0.0001
    G_optimizer = torch.optim.Adam(G.parameters(), G_lr, betas = (0.5, 0.999))
    D_optimizer = torch.optim.Adam(D.parameters(), D_lr, betas = (0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss(reduction = 'mean')

    G_loss_list = []
    D_loss_list = []

    for epoch in range(epochs):

        G.train()
        D.train()

        epoch_G_loss = 0.0
        epoch_D_loss = 0.0

        print('-------------')
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        for images, _ in dataloader:
            images = images.to(device)
            batch_size = images.size()[0]

            real_label = torch.full((batch_size, ), 1).to(device)
            fake_label = torch.full((batch_size, ), 0).to(device)

            # -----Discriminatorの学習-----
            # 真の画像を入力して識別
            D_out_real, _ = D(images)

            # 偽の画像を入力して識別
            random_noise = torch.randn(batch_size, z_dim).to(device)
            fake_images = G(random_noise)
            D_out_fake, _ = D(fake_images)

            # long型からTensor型への変換
            real_label = real_label.type_as(D_out_real.view(-1))
            fake_label = fake_label.type_as(D_out_fake.view(-1))

            # 誤差の計算
            D_loss_real = criterion(D_out_real.view(-1), real_label)
            D_loss_fake = criterion(D_out_fake.view(-1), fake_label)
            D_loss = D_loss_real + D_loss_fake

            # 重みの更新
            G_optimizer.zero_grad()
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()
            
            # -----Generatorの学習-----
            # 偽の画像を入力して識別
            random_noise = torch.randn(batch_size, z_dim).to(device)
            fake_images = G(random_noise)
            D_out_fake, _ = D(fake_images)
            # 誤差の計算
            # 生成した画像が本物に近づくように学習
            G_loss = criterion(D_out_fake.view(-1), real_label)
            G_optimizer.zero_grad()
            D_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            epoch_G_loss += G_loss.item()
            epoch_D_loss += D_loss.item()

        G_loss_list.append(epoch_G_loss / batch_size)
        D_loss_list.append(epoch_D_loss / batch_size)
        print("Generator Loss : {:.4f} ||Discriminator Loss : {:.4f}".format(epoch_G_loss / batch_size, epoch_D_loss / batch_size))

        z = torch.randn(32, z_dim)

        G.eval()
        fake_images = G(z.to(device))
        
        for i in range(len(fake_images)):
            plt.subplot(4, 8, i + 1)
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            plt.imshow(fake_images[i][0].cpu().detach().numpy(), cmap = "gray")
        plt.savefig("./train_generate_img/AnoGAN_mnist_" + "{0:03d}".format(epoch) + ".png")
        plt.clf()

    return G, D, G_loss_list, D_loss_list

if __name__ == '__main__':

    os.chdir("/src/AnoGAN/mnist")

    Epoch = 100

    update_G, update_D, generator_loss, discriminator_loss = train(train_loader, Epoch)

    torch.save(update_G.state_dict(), "./AnoGAN_mnist_generator.pth")
    torch.save(update_D.state_dict(), "./AnoGAN_mnist_discriminator.pth")

    plt.plot(range(Epoch), generator_loss, c = "orangered", label = "Generator")
    plt.plot(range(Epoch), discriminator_loss, c = "royalblue", label = "Discriminator")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.xlim(0, Epoch)
    plt.savefig("./loss_graph.png")