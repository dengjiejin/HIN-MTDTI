import torch
import torch.nn as nn
import torch.nn.functional as F
from gat_layers import GraphAttentionLayer, SpGraphAttentionLayer
from torch_geometric.nn import GATConv, GCNConv
from torch.nn.utils.weight_norm import weight_norm
from ban import BANLayer
from torch_geometric import transforms

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout, negative_slope, nheads):
        """ version of GAT."""
        super(GAT, self).__init__()
        # self.dropout = 0.6
        self.dropout = dropout
        self.attentions = GATConv(nfeat, nhid, nheads, True, negative_slope=negative_slope, dropout=self.dropout)
        self.out_att = GATConv(nhid*nheads, noutput, 1, False, negative_slope=negative_slope, dropout=self.dropout)
        self.BatchNorm = torch.nn.BatchNorm1d(num_features=noutput)
        self.LayerNorm = torch.nn.LayerNorm(noutput)

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.attentions(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, edge_index))
        x = self.BatchNorm(x)
        x = self.LayerNorm(x)
        return x

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout):
        """version of GCN."""
        super(GCN, self).__init__()
        # self.dropout = 0.6
        self.dropout = dropout
        self.gcn1 = GCNConv(nfeat, noutput)
        self.gcn1_1 = GCNConv(nhid,nhid)
        self.gcn1_2 = GCNConv(nhid,nhid)
        self.gcn1_3 = GCNConv(nhid,nhid)

        self.gcn2 = GCNConv(nhid, noutput)
        self.BatchNorm = torch.nn.BatchNorm1d(num_features=noutput)
        self.LayerNorm = torch.nn.LayerNorm(noutput)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.gcn1(x, edge_index, edge_weight))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.gcn1_1(x, edge_index, edge_weight))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.gcn1_2(x, edge_index, edge_weight))
       # x = F.dropout(x, self.dropout, training=self.training)
       # x = F.elu(self.gcn1_3(x, edge_index, edge_weight))

        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.gcn2(x, edge_index, edge_weight))
        x = self.BatchNorm(x)
        x = self.LayerNorm(x)
        return x

class NN(nn.Module):
    def __init__(self, ninput, nhidden, noutput, nlayers, dropout=0.3):
        """
        """
        super(NN, self).__init__()
        self.dropout = dropout
        self.encode = torch.nn.ModuleList([
            torch.nn.Linear(ninput if l == 0 else nhidden[l - 1], nhidden[l] if l != nlayers - 1 else noutput) for l in
            range(nlayers)])
        self.BatchNormList = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=nhidden[l] if l != nlayers-1 else noutput) for l in range(nlayers)])
        self.LayerNormList = torch.nn.ModuleList([
            torch.nn.LayerNorm(nhidden[l] if l != nlayers - 1 else noutput) for l in range(nlayers)])

    def forward(self, x):
        # x [B, 220] or [B, 881]
        for l, linear in enumerate(self.encode):
            x = F.relu(linear(x))
            x = self.BatchNormList[l](x)
            x = self.LayerNormList[l](x)
            x = F.dropout(x, self.dropout)
        return x

class DTI_Decoder(nn.Module):
    def __init__(self, Protein_num, Drug_num, Nodefeat_size, nhidden, nlayers, dropout=0.3):
        super(DTI_Decoder, self).__init__()
        self.Protein_num = Protein_num
        self.Drug_num = Drug_num
        self.dropout = dropout
        self.nlayers = nlayers
        self.decode = torch.nn.ModuleList([
            torch.nn.Linear(Nodefeat_size if l == 0 else nhidden[l - 1], nhidden[l]) for l in
            range(nlayers)])
        self.BatchNormList = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=nhidden[l]) for l in range(nlayers)])
        self.linear = torch.nn.Linear(nhidden[nlayers-1], 1)
        self.bcn = weight_norm(
            BANLayer(256, 256, h_dim=256, h_out=2),
            name='h_mat', dim=None)
        self.li = torch.nn.Linear(256*3, 256)
        self.attention = nn.Parameter(torch.ones(3, 1, 1) / 3)

    def forward(self, nodes_features, protein_index, drug_index, protein_ori, drug_ori):
        # protein id index in node id
        # drug id index in node id
        protein_features = nodes_features[protein_index]
        drug_features = nodes_features[drug_index]
        # pair_nodes_features = protein_features*drug_features
        ######################################################
        drug_features_new = drug_features.unsqueeze(1)
        protein_features_new = protein_features.unsqueeze(1)
        pair_nodes_features = self.bcn(drug_features_new, protein_features_new)[0]  # 4680 256
        #
        attention = torch.softmax(self.attention, dim=0)
        l = [pair_nodes_features, protein_ori, drug_ori]
        xc = torch.stack(l, dim=0)
        pair_nodes_features = torch.sum(xc * attention, dim=0)
        #########################################################################
        # pair_nodes_features = torch.cat((pair_nodes_features, protein_ori, drug_ori), dim=1)
        # pair_nodes_features = self.li(pair_nodes_features)

        for l, dti_nn in enumerate(self.decode):
            pair_nodes_features = F.dropout(pair_nodes_features, self.dropout)
            pair_nodes_features = F.relu(dti_nn(pair_nodes_features))
            pair_nodes_features = self.BatchNormList[l](pair_nodes_features)
        pair_nodes_features = F.dropout(pair_nodes_features, self.dropout)
        output = self.linear(pair_nodes_features)
        return torch.sigmoid(output)

class DTI_Graph(nn.Module):
    """
    Model for Drug-Protein interaction Graph
    pnn_hyper = [protein_ninput, pnn_nhid, gat_ninput, pnn_nlayers]
    dnn_hyper = [drug_ninput, dnn_nhid, gat_ninput, dnn_nlayers]
    GAT_hyper = [gat_ninput, gat_nhid, gat_noutput, gat_negative_slope, nheads]
    Deco_hyper = [gat_noutput, DTI_nn_nhid]
    """
    def __init__(self, GAT_hyper, PNN_hyper, DNN_hyper, DECO_hyper, Protein_num, Drug_num, dropout):
        super(DTI_Graph, self).__init__()
        self.drug_nn = NN(DNN_hyper[0], DNN_hyper[1], DNN_hyper[2], DNN_hyper[3], dropout)
        self.protein_nn = NN(PNN_hyper[0], PNN_hyper[1], PNN_hyper[2], PNN_hyper[3], dropout)
        self.gat = GAT(GAT_hyper[0], GAT_hyper[1], GAT_hyper[2], dropout, GAT_hyper[3], GAT_hyper[4])
        self.gcn = GCN(GAT_hyper[0], GAT_hyper[1], GAT_hyper[2], dropout)
        self.DTI_Decoder = DTI_Decoder(Protein_num, Drug_num, DECO_hyper[0], DECO_hyper[1], DECO_hyper[2], dropout)
        self.Protein_num = Protein_num
        self.Drug_num = Drug_num
        self.BatchNorm = torch.nn.BatchNorm1d(num_features=GAT_hyper[0])
        self.LayerNorm = torch.nn.LayerNorm(GAT_hyper[0])

    def forward(self, Proteins, Drugs, edge_index, protein_index, drug_index, edge_weight):

        # Protein and Drug embeding
        emb_drugs = self.drug_nn(Drugs)

        emb_proteins = self.protein_nn(Proteins)
        # emb_proteins nxi, emb_drugs mxi
        Nodes_features = torch.cat((emb_proteins, emb_drugs), 0)
        Nodes_features = self.BatchNorm(Nodes_features)
        Nodes_features = self.LayerNorm(Nodes_features)
        protein_ori = Nodes_features[protein_index]     # 4680 256
        drug_ori = Nodes_features[drug_index]   # 4680 256
        # gat
        # Nodes_features = self.gat(Nodes_features, edge_index)
        #gcn
        Nodes_features = self.gcn(Nodes_features, edge_index, edge_weight)
        # Decoder
        output = self.DTI_Decoder(Nodes_features, protein_index, drug_index, protein_ori, drug_ori)
        output = output.view(-1)
        return output