import torch
import torch.nn as nn
import torch.nn.functional as F


# LSTM model for weights dynamic adaptation
class AdaptiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AdaptiveLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # Initialize weights with predefined values
        # self.fc.weight.data = torch.tensor([[10.0, 10.0, 5.0, 5.0]])  # Adjust based on your specific initial weights
        # self.fc.bias.data = torch.zeros(output_size)
        # Initialize LSTM parameters with an identity initialization
        # self.init_lstm_weights()

    def init_lstm_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.eye_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # lstm_out = lstm_out[:, -1, :]  # Take the output from the last time step
        # output = F.sigmoid(self.fc(lstm_out))
        # output = F.softmax(self.fc(lstm_out)) + 10 ** -3
        output = F.sigmoid(self.fc(lstm_out)) + 10 ** - 2
        return output


class CoeffOfVariation(nn.Module):
    """
        Wrapper of the BaseLoss which weighs the losses to the Cov-Weighting method,
        where the statistics are maintained through Welford's algorithm.
    """

    def __init__(self, opt, device):
        super(CoeffOfVariation, self).__init__()

        # How to compute the mean statistics: Full mean or decaying mean.
        self.mean_decay = True if opt.mean_sort == 'decay' else False
        self.mean_decay_param = opt.mean_decay_param
        self.num_losses = 4
        self.current_iter = -1


        # Initialize all running statistics at 0.
        self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=True).type(torch.FloatTensor).to(device)
        self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=True).type(torch.FloatTensor).to(device)
        self.alphas_start = torch.tensor([torch.tensor(0.25), torch.tensor(0.25), torch.tensor(0.25), torch.tensor(0.25)], requires_grad=True).type(torch.FloatTensor).to(device)
        print(self.alphas_start)
        self.running_S_l = torch.zeros((self.num_losses,), requires_grad=True).type(torch.FloatTensor).to(device)
        self.running_std_l = None

    def forward(self, L):

        # Increase the current iteration parameter.
        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L if self.current_iter == 0 else self.running_mean_L   #  .clone()
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        print(self.current_iter)
        # If we are in the first iteration set alphas to all 1/32
        if self.current_iter <= 2:
            alphas = self.alphas_start
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        elif self.current_iter > 0 and self.mean_decay:
            mean_param = self.mean_decay_param
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))

        # 2. Update the statistics for l
        x_l = l  #  .clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L  #  .clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        return alphas, self.current_iter


if __name__ == "__main__":
    adaptive_lstm = AdaptiveLSTM(input_size=64*64, hidden_size=128, output_size=4)

    # weights = adaptive_lstm(loss_terms.unsqueeze(0))
    # print(weights)
    loss_terms = torch.stack([torch.tensor(0), torch.tensor(3.4), torch.tensor(2.5), torch.tensor(0)], dim=-1)
    weights = adaptive_lstm(loss_terms.unsqueeze(0))
    print(weights)
