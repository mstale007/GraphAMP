import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        GINConv(Sequential(Linear(78, 32), ReLU(), Linear(32, 32)))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)

        # cell line feature
        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)
        self.fc1_xt = nn.Linear(2144, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.n11=GINConv(Sequential(Linear(78, 32), ReLU(), Linear(32, 32)))
        self.n12=GINConv(Sequential(Linear(32, 32), ReLU(), Linear(32, 32)))
        self.n13=GINConv(Sequential(Linear(32, 32), ReLU(), Linear(32, 32)))
        
        self.n2=Linear(32, 128)
        self.n31=Linear(208, 170)
        self.n32=Linear(170, 128)
        self.n33=Linear(208, 128)
        self.n4=Linear(2*128, 128)
        self.n5=Linear(128, 1)
        self.bn1 = torch.nn.BatchNorm1d(208)
        self.bn2 = torch.nn.BatchNorm1d(32)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        #print(x)
        #print(data.target)
        # x = F.relu(self.n11(x, edge_index))
        # x = torch.nn.BatchNorm1d(32)(x)
        # x = F.relu(self.n12(x, edge_index))
        # x = torch.nn.BatchNorm1d(32)(x)
        # x = F.relu(self.n13(x, edge_index))
        # x = torch.nn.BatchNorm1d(32)(x)

        x = F.elu(self.n11(x, edge_index))
        x = torch.nn.BatchNorm1d(32)(x)
        x = F.relu(self.n12(x, edge_index))
        x = torch.nn.BatchNorm1d(32)(x)
        x = F.relu(self.n13(x, edge_index))
        x = torch.nn.BatchNorm1d(32)(x)
        # x = F.relu(self.conv3(x, edge_index))
        # x = self.bn3(x)
        # x = F.relu(self.conv4(x, edge_index))
        # x = self.bn4(x)
        # x = F.relu(self.conv5(x, edge_index))
        # x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.n2(x))
        x = F.dropout(x, p=0.2, training=self.training)

        # protein input feed-forward:
        target = data.target

        # 1d conv layers
        # conv_xt = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8)(target)
        # target = target[:,None,:]
        # conv_xt = self.conv_xt_1(target)
        # conv_xt = F.relu(conv_xt)
        # conv_xt = self.pool_xt_1(conv_xt)
        # conv_xt = self.conv_xt_2(conv_xt)
        # conv_xt = F.relu(conv_xt)
        # conv_xt = self.pool_xt_2(conv_xt)
        # conv_xt = self.conv_xt_3(conv_xt)
        # conv_xt = F.relu(conv_xt)
        # conv_xt = self.pool_xt_3(conv_xt)

        # # flatten
        # xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        # xt = self.fc1_xt(xt)
        target = self.bn1(target)
        conv_xt = self.n31(target)
        conv_xt = self.n32(conv_xt)
        conv_xt = F.softmax(conv_xt)

        # conv_xt = self.pool_xt_1(conv_xt)
        # conv_xt = self.conv_xt_2(conv_xt)
        # conv_xt = F.relu(conv_xt)
        # conv_xt = self.pool_xt_2(conv_xt)
        # conv_xt = self.conv_xt_3(conv_xt)
        # conv_xt = F.relu(conv_xt)
        # conv_xt = nn.MaxPool1d(3)(conv_xt)
        
        # flatten
        xt = nn.Flatten()(conv_xt)
        # xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])

        
        
        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.n4(xc)
        xc = ReLU()(xc)
        xc = nn.Dropout(0.5)(xc)
        # xc = self.fc2(xc)
        # xc = self.relu(xc)
        # xc = self.dropout(xc)
        out = self.n5(xc)
        out = nn.Sigmoid()(out)
        return out, x
