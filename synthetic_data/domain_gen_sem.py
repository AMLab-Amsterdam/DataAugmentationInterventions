import torch


class DomainGenToyData(object):
    def __init__(self,
                 dim,
                 transform_c=False,
                 transform_y_and_d=False,
                 additional_noise_y_and_d=False,
                 additional_noise_x1_and_x2=False):

        self.transform_c = transform_c
        self.transform_y_and_d = transform_y_and_d
        self.additional_noise_y_and_d = additional_noise_y_and_d
        self.additional_noise_x1_and_x2 = additional_noise_x1_and_x2

        # dim is the number of dimensions of x
        self.dim_half = dim//2

        # Linear transformation c to y and d
        if transform_c:
            self.Wcd = torch.randn(self.dim_half, self.dim_half) / dim
            self.Wcy = torch.randn(self.dim_half, self.dim_half) / dim
        else:
            self.Wcd = torch.eye(self.dim_half)
            self.Wcy = torch.eye(self.dim_half)

        # Linear transformation from d to x1 and y to x2
        if transform_y_and_d:
            self.Wdx1 = torch.randn(self.dim_half, self.dim_half) / dim
            self.Wyx2 = torch.randn(self.dim_half, self.dim_half) / dim
        else:
            self.Wdx1 = torch.eye(self.dim_half)
            self.Wyx2 = torch.eye(self.dim_half)

    def __call__(self, N, train=True):

        # Add noise to y and d
        if self.additional_noise_y_and_d:
            noise_y = torch.randn(N, self.dim_half)*0.1
            noise_d = torch.randn(N, self.dim_half)*0.1
        else:
            noise_y = 0
            noise_d = 0

        # Add noise to x1 and x2
        if self.additional_noise_x1_and_x2:
            noise_x1 = torch.randn(N, self.dim_half)*0.1
            noise_x2 = torch.randn(N, self.dim_half)*0.1
        else:
            noise_x1 = 0
            noise_x2 = 0

        # Sample values for confounder
        c = torch.randn(N, self.dim_half)

        # Compute d
        if train:
            d = c @ self.Wcd + noise_d

        else:
            d = torch.randn(N, self.dim_half)

        # Compute y
        y = c @ self.Wcy + noise_y

        # Compute x1 and x2
        x1 = d @ self.Wdx1 + noise_x1
        x2 = y @ self.Wyx2 + noise_x2

        return torch.cat((x1, x2), dim=1).numpy(), y.sum(dim=1).numpy(), d.sum(dim=1).numpy()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    number_of_samples = 1000
    dim = 10

    data_loader = DomainGenToyData(dim,
                                    transform_c=False,
                                    transform_y_and_d=False,
                                    additional_noise_y_and_d=True,
                                    additional_noise_x1_and_x2=True)

    x, y, d = data_loader(number_of_samples, train=True)

    plt.figure()
    plt.scatter(y, d)
    plt.xlabel('$y$')
    plt.ylabel('$d$')
    plt.savefig('load_domain_gen_toy_train_FFTT.png', bbox_inches='tight')


    x, y, d = data_loader(number_of_samples, train=False)

    plt.figure()
    plt.scatter(y, d)
    plt.xlabel('$y$')
    plt.ylabel('$d$')
    plt.savefig('load_domain_gen_toy_test_FFTT.png', bbox_inches='tight')