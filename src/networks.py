import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Generator(nn.Module):
    def __init__(self, nz):

        super(Generator, self).__init__()

        def block (n_in_channels, n_out_channels, batch_norm=True):
            layers = []
            layers.append(nn.Linear(n_in_channels, n_out_channels))
            if batch_norm:
                layers.append(nn.BatchNorm1d(n_out_channels, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.main = nn.Sequential(
            *block(nz, 256, batch_norm=False),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        fake_image = self.main(z)
        return fake_image.view(fake_image.size(0), 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):

        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)