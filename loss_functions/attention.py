import torch.nn as nn
import torch
from matplotlib import pyplot as plt


class Self_AttnRes(nn.Module):
    """ Self attention Layer with resampling customisation"""

    # todo

    def __init__(self, in_dim, out_dim, activation='relu'):
        super(Self_AttnRes, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # nn.Parameter is a convenient way to define a learnable parameters;

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        print(attention.shape)
        # Resampling layer
        # element wise product between attention and x

        # -------
        # to modify
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        # -------

        out = self.gamma * out + x * 0.001

        return out, attention


class Self_Attn(nn.Module):
    """Self attention Layer"""

    def __init__(self, in_dim, activation='relu'):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # nn.Parameter is a convenient way to define a learnable parameters;

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        print(proj_query.shape)
        print(proj_key.shape)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        print(energy.shape)
        attention = self.softmax(energy)  # BX (N) X (N)
        print(attention.shape)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        # print("gamma:")
        # print(self.gamma)
        # print("gamma*out:")
        # print(self.gamma * out)
        out = self.gamma * out + x
        print(out.shape)
        return out, attention, self.gamma


if __name__ == '__main__':
    import torch.nn.functional as F

    """loss_terms = torch.stack([torch.tensor(1., requires_grad=True), torch.tensor(1.,requires_grad=True), torch.tensor(0.5, requires_grad=True), torch.tensor(0.5, requires_grad=True)], dim=0).unsqueeze(0).unsqueeze(1).unsqueeze(2)
    print(loss_terms)
    mask_idA = torch.tensor([[[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]])
    print(mask_idA.shape)
    print((loss_terms*mask_idA).sum())
    tensor_with_grad = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    # Normalize the tensor
    normalized_x = F.normalize(loss_terms, p=2, dim=-1)+0.1
    print(normalized_x)
    # Get the norms of the original tensor
    norms = torch.norm(loss_terms, p=2, dim=-1)
    print(norms)
    # Revert back to the unnormalized values
    unnormalized_x = normalized_x * norms.view(-1, 1)
    print(unnormalized_x)"""

    attention = Self_Attn(1, 'relu')
    tensor = torch.empty((16, 1, 4, 4))
    # tensor = tensor[:, 0, :, :]
    # print(tensor.shape)
    out, att, _ = attention(tensor)

