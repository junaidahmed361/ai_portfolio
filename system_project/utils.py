import pandas as pd
import numpy as np
import anndata as ad
from sklearn.preprocessing import LabelEncoder
from typing import List, Optional, Callable
import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.data import Data, download_url
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATv2Conv,
    SAGEConv,
    GINConv,
    global_mean_pool,
    global_add_pool,
)
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_predictions(loader, model_, val=False):
    val_preds = []
    val_labels = []
    test_preds = []
    test_labels = []
    # confusion matrix
    # print(f'Confusion Matrix: {confusion_matrix(y_test,y_pred)}')
    if val:
        for batch in loader:
            batch = batch.to(device)
            pred = model_(batch.to(device))
            _, predicted = torch.max(pred.data, 1)
            val_preds.append(predicted)
            val_labels.append(batch.y)

        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)

        print(
            f"Validation accuracy: {accuracy_score(val_labels.cpu(),val_preds.cpu())*100}"
        )

        print(
            f"Validation F1: {f1_score(val_labels.cpu(),val_preds.cpu(),average='weighted')*100}"
        )

    else:
        for batch in loader:
            batch = batch.to(device)
            pred = model_(batch.to(device))
            _, predicted = torch.max(pred.data, 1)
            test_preds.append(predicted)
            test_labels.append(batch.y)

        test_preds = torch.cat(test_preds)
        test_labels = torch.cat(test_labels)

        # Accuracy
        print(
            f"Test accuracy: {accuracy_score(test_labels.cpu(),test_preds.cpu())*100}"
        )

        # F1 score
        print(
            f"Test F1: {f1_score(test_labels.cpu(),test_preds.cpu(),average='weighted')*100}"
        )

        # Classification report
        print(
            f"Classification Report: \n{classification_report(test_labels.cpu(),test_preds.cpu())}"
        )

    return test_preds, test_labels


def anndata_to_pygdata(
    dataset: ad.AnnData,
    position_keys: List[str],
    sample_keys: List[str],
    target_key: Optional[str] = None,
    covariate_keys: Optional[List[str]] = None,
    feature_norm_key: Optional[str] = None,
):
    """Convert spatial transcriptomics data stored in an AnnData structure to a
    collection of PyTorch Geometric graphs."""

    dataset.obs.reset_index(inplace=True)
    dataset.obs.index = pd.RangeIndex(len(dataset.obs.index))

    if target_key is not None:
        le = LabelEncoder()
        le.fit(dataset.obs[target_key])

        dataset.obs["new_target"] = le.transform(dataset.obs[target_key])

        obs = dataset.obs.groupby(sample_keys)

        y = dataset.obs["new_target"].values
    else:
        y = None
    graphs = []
    for group in obs:
        graph_name = group[0]
        graph_df = group[1]
        graph_nodes = graph_df.index
        pos = torch.tensor(graph_df[position_keys].to_numpy())
        graph_x = dataset[graph_nodes].X
        if feature_norm_key:
            feature_norm = dataset.var[feature_norm_key].to_numpy()
            graph_x = graph_x / feature_norm.reshape((1, -1))
        graph_x = torch.tensor(graph_x, dtype=torch.float32)
        if y is not None:
            graph_y = torch.tensor(y[graph_nodes])
        else:
            graph_y = None
        if covariate_keys is not None:
            cov_dict = {
                key: torch.tensor(graph_df[key].to_numpy()) for key in covariate_keys
            }
        else:
            cov_dict = dict()

        graph = Data(x=graph_x, y=graph_y, pos=pos, sample_name=graph_name, **cov_dict)
        graphs.append(graph)
    return graphs


def spatial_transcriptomics_graph_dataset_factory(
    dataset_url: str,
    raw_filename: str,
    position_keys: List[str],
    sample_keys: List[str],
    target_key: Optional[str] = None,
    covariate_keys: Optional[List[str]] = None,
    feature_norm_key: Optional[str] = None,
) -> Callable:
    """Generates a PyTorch Geometric InMemoryDataset class from a given source
    url for an AnnData array and selected keys that identify spot positions,
    sample images, and spotwise targets."""

    class STDataset(tg.data.InMemoryDataset):
        """Dataset of spatial transcriptomics spots."""

        def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])

        @property
        def raw_file_names(self):
            return [raw_filename]

        @property
        def processed_file_names(self):
            return ["data.pt"]

        def download(self):
            download_url(dataset_url, self.raw_dir)

        def get_edge_attr(self, data):
            num_edges = data.num_edges
            # num_edge_features = data.num_edge_features
            num_edge_features = 2 * data.num_features + 4

            print(
                f"Graph {data.sample_name} has {data.num_nodes} nodes, {num_edges} edges, and {num_edge_features} edge features."
            )

            left_nodes = data.edge_index[0, :]
            right_nodes = data.edge_index[1, :]
            left_node_attr = data.x[left_nodes, :]  # (n_edges, n_node_attr)
            right_node_attr = data.x[right_nodes, :]
            left_node_pos = data.pos[left_nodes, :]  # (n_edges, 2)
            right_node_pos = data.pos[right_nodes, :]
            edge_attr = torch.cat(
                (left_node_attr, right_node_attr, left_node_pos, right_node_pos), dim=1
            )
            edge_attr = edge_attr.type(torch.float32)

            data.edge_attr = edge_attr

            return data

        def process(self):
            dataset = ad.read(self.raw_paths[0])
            data_list = anndata_to_pygdata(
                dataset,
                position_keys,
                sample_keys,
                target_key,
                covariate_keys,
                feature_norm_key,
            )

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
                # get edge attrs
                data_list = [self.get_edge_attr(data) for data in data_list]
            train_flag = torch.rand(len(data_list)) < 0.8

            for i in range(len(data_list)):
                data_list[i]["train"] = train_flag[i]
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

    return STDataset


class Learner(nn.Module):
    def __init__(self, model, data_loader, optimizer, loss_fn, device="cpu"):
        super(Learner, self).__init__()
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.losses = []
        self.accuracies = []
        self.device = device

    def train(self):
        size = len(self.data_loader)
        for i, batch in enumerate(self.data_loader):
            # Compute prediction error
            batch = batch.to(device)
            pred = self.model(batch.to(device))
            loss = self.loss_fn(pred, batch.y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.losses.append(loss.item())
            if i % 100 == 0:
                loss, current = loss.item(), i * len(batch.x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                self.losses.append(loss)

        # print train and validation accuracy
        _, predicted = torch.max(pred.data, 1)
        correct = (predicted == batch.y).sum().item()
        accuracy = correct / len(batch.y)
        print(f"Train accuracy: {accuracy*100}")
        self.accuracies.append(accuracy)

        # print train loss
        print(f"Train loss: {loss}")

    # get validation accuracy
    def validate(self, val_loader):
        size = len(val_loader)
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                batch = batch.to(device)
                pred = self.model(batch.to(device))
                _, predicted = torch.max(pred.data, 1)
                correct += (predicted == batch.y).sum().item()
        print(f"Validation accuracy: {correct/size}")

    def test(self, test_loader):
        size = len(test_loader.dataset)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = self.model(batch)
                test_loss += self.loss_fn(pred, batch.y).item()
                _, predicted = torch.max(pred.data, 1)
                correct += (predicted == batch.y).sum().item()
        test_loss /= size
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    def plot_losses(self):
        plt.plot(self.losses)
        plt.title("Losses")
        plt.show()

    def plot_accuracies(self):
        plt.plot(self.accuracies)
        plt.title("Accuracies")
        plt.show()

    def run(self, epochs, val_loader=None):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train()
            if val_loader:
                self.validate(val_loader)
        print("Done!")
        # self.test()
        print("Done!")

    def reset_all_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def reset_weights(self, layer):
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()


def plot_correct_preds(model_, data_loader, device="cpu", model_name: str = None):
    # use seaborn to plot the correct predictions only in green and the incorrect in red
    fig, axs = plt.subplots(nrows=7, ncols=3, figsize=(8, 6))
    fig.suptitle(f"Predicted vs Actual: {model_name}")
    for batch, ax in zip(data_loader, axs.ravel()):
        # plot the prediction against the positions
        batch = batch.to(device)
        pred = model_(batch.to(device))
        _, predicted = torch.max(pred.data, 1)
        # plot with legend where colors are the predicted labels
        ax.scatter(
            batch.pos[:, 0].cpu(), batch.pos[:, 1].cpu(), c=predicted.cpu().numpy()
        )

        # plot the correct predictions in green
        correct = (predicted == batch.y).squeeze()
        ax.scatter(
            batch.pos[correct, 0].cpu(),
            batch.pos[correct, 1].cpu(),
            c="limegreen",
            label="Correct",
        )

        # plot the incorrect predictions in red
        incorrect = (predicted != batch.y).squeeze()
        ax.scatter(
            batch.pos[incorrect, 0].cpu(),
            batch.pos[incorrect, 1].cpu(),
            c="tomato",
            label="Incorrect",
        )

        ax.label_outer()

    # show the legend in the top right corner off the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.show()


def plot_preds(model_, data_loader, device="cpu", show_all=False):
    fig1, axs1 = plt.subplots(nrows=7, ncols=3, figsize=(8, 6))
    # fig1.suptitle('Predicted')
    for batch, ax in zip(data_loader, axs1.ravel()):
        # plot the prediction against the positions
        batch = batch.to(device)
        pred = model_(batch.to(device))
        _, predicted = torch.max(pred.data, 1)
        # plot with legend where colors are the predicted labels
        ax.scatter(
            batch.pos[:, 0].cpu(), batch.pos[:, 1].cpu(), c=predicted.cpu().numpy()
        )

        ax.label_outer()

    # create a new figure for the true labels
    fig2, axs2 = plt.subplots(nrows=7, ncols=3, figsize=(8, 6))
    # fig2.suptitle('True')
    for batch, ax in zip(data_loader, axs2.ravel()):
        # plot the prediction against the positions
        batch = batch.to(device)
        pred = model_(batch.to(device))
        _, predicted = torch.max(pred.data, 1)
        # plot with legend where colors are the predicted labels
        ax.scatter(
            batch.pos[:, 0].cpu(), batch.pos[:, 1].cpu(), c=batch.y.cpu().numpy()
        )

        ax.label_outer()

    # present both figures as subplots in one figure with 2 columns and 1 row
    nrow, ncol = 1, 2
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8, 6),
        gridspec_kw=dict(
            wspace=0.0,
            hspace=0.0,
            top=1.0 - 0.5 / (nrow + 1),
            bottom=0 / (nrow + 1),
            left=0 / (ncol + 1),
            right=1 - 0.5 / (ncol + 1),
        ),
        sharex=True,
        sharey=True,
    )
    # fig.suptitle('Predicted vs True')
    fig.tight_layout()

    # draw the figures
    fig1.canvas.draw()
    fig2.canvas.draw()

    # convert the figures to numpy arrays
    data1 = np.fromstring(fig1.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data2 = np.fromstring(fig2.canvas.tostring_rgb(), dtype=np.uint8, sep="")

    data1 = data1.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data2 = data2.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # add the numpy arrays to the figure
    axs[0].imshow(data1)
    axs[0].set_title("Predicted")
    axs[1].imshow(data2)
    axs[1].set_title("True")

    [axi.set_axis_off() for axi in axs.ravel()]

    if show_all:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)

    # expand fig
    fig.set_size_inches(16, 6)

    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace=0, wspace=0)
    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.show()


# create GCN model class with 2 hidden layers and 1 output layer
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 32)  # [100x100] x []
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)


class SpatialKernel(nn.Module):
    """This is a wrapper around GCNConv that allows us to use a kernel function to compute the edge weights."""

    def __init__(self, in_channels, out_channels):
        super(SpatialKernel, self).__init__()
        self.sigma = 1000  # eventually this should be a learnable parameter
        # self.k = self.compute_kernel()
        # print(self.k)
        self.ll1 = nn.Linear(in_channels, 32)  # [100x100] x []
        self.ll2 = nn.Linear(32, 32)
        self.ll3 = nn.Linear(32, out_channels)

    def gaussian_kernel(self, dist_matrix, sigma_constant):
        return torch.exp(-(dist_matrix**2) * sigma_constant)
        # return torch.exp(-dist_matrix**2 / self.sigma**2)

    def compute_kernel(self, pos):
        # compute distance matrix
        sigma_constant = 1 / (2 * self.sigma**2)
        kernel = np.zeros((pos.shape[0], pos.shape[0]))

        # print('computing kernel')

        dist = torch.cdist(pos, pos)
        # print(dist)
        kernel = self.gaussian_kernel(torch.tensor(dist), sigma_constant)

        return kernel

    def forward(self, data):
        # KxW
        # [NxN] x [NxD] x [DxE] = [NxE] => [232, 32]

        # multiple by weight matrix
        # x = torch.matmul(x, self.weight) # [N, 32]

        # multiply kernel by x
        # x = torch.matmul(self.kernel, x) # [N, 32]

        kernel = self.compute_kernel(data.pos)
        x = data.x

        # convert kernel to tensor of type float
        kernel = torch.tensor(kernel, dtype=torch.float)

        # multiply kernel by x --> at each layer...
        x = torch.matmul(kernel, x)  # [N, 32]
        x = self.ll1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = torch.matmul(kernel, x)  # [32, 32]
        x = self.ll2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = torch.matmul(kernel, x)  # [32, 32]
        x = self.ll3(x)

        return F.log_softmax(x, dim=1)


class SpatialGCN(nn.Module):
    """This is a wrapper around GCNConv that allows us to use a kernel function to compute the edge weights."""

    def __init__(self, in_channels, out_channels):
        super(SpatialGCN, self).__init__()
        self.sigma = 1000  # eventually this should be a learnable parameter
        self.ll1 = GCNConv(in_channels, 32)  # [100x100] x []
        self.ll2 = GCNConv(32, 32)
        self.ll3 = GCNConv(32, out_channels)

    def gaussian_kernel(self, dist_matrix, sigma_constant):
        return torch.exp(-(dist_matrix**2) * sigma_constant)

    def compute_kernel(self, pos):
        # compute distance matrix
        sigma_constant = 1 / (2 * self.sigma**2)
        kernel = np.zeros((pos.shape[0], pos.shape[0]))

        dist = torch.cdist(pos, pos)
        kernel = self.gaussian_kernel(torch.tensor(dist), sigma_constant)

        return kernel

    def forward(self, data):
        # KxW
        # [NxN] x [NxD] x [DxE] = [NxE] => [232, 32]

        # multiple by weight matrix
        # x = torch.matmul(x, self.weight) # [N, 32]

        # multiply kernel by x
        # x = torch.matmul(self.kernel, x) # [N, 32]

        kernel = self.compute_kernel(data.pos)
        x = data.x
        edge_index = data.edge_index

        # convert kernel to tensor of type float
        kernel = torch.tensor(kernel, dtype=torch.float)

        # multiply kernel by x --> at each layer...
        x = torch.matmul(kernel, x)  # [N, 32]
        x = self.ll1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = torch.matmul(kernel, x)  # [32, 32]
        x = self.ll2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = torch.matmul(kernel, x)  # [32, 32]
        x = self.ll3(x, edge_index)

        return F.log_softmax(x, dim=1)


class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


class NNConv_old(MessagePassing):
    """The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),
    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
            :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        nn,
        aggr="add",
        root_weight=True,
        bias=True,
        **kwargs,
    ):
        super(NNConv_old, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter("root", None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        size = self.in_channels
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo.float())

    def message(self, x_j, pseudo):
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


class GraphPDE(torch.nn.Module):
    # width_kernel = static (how wide HLs are)
    # depth = static (how many CLs)
    # ker_in = number of edge features (2*#node features + 4)
    # out_width = 3
    # in_width = 232
    # width_node = 64, ...
    def __init__(self, width_kernel, depth, ker_in, in_width=1, out_width=1):
        super(GraphPDE, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, 32)

        kernel1 = DenseNet(
            [
                ker_in,
                width_kernel // 4,
                width_kernel // 2,
                width_kernel,
                width_kernel,
                32 * 16,
            ],
            torch.nn.ReLU,
        )
        # width node = in/out features/node (e.g. 1 feature per node)
        self.conv1 = NNConv_old(32, 16, kernel1, aggr="mean")

        kernel2 = DenseNet(
            [
                ker_in,
                width_kernel // 4,
                width_kernel // 2,
                width_kernel,
                width_kernel,
                16 * 16,
            ],
            torch.nn.ReLU,
        )
        self.conv2 = NNConv_old(16, 16, kernel2, aggr="mean")

        self.fc2 = torch.nn.Linear(16, out_width)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    """Graph Attention Network"""

    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h * heads, dim_out, heads=1)
        # self.optimizer = torch.optim.Adam(self.parameters(),
        #                                   lr=0.005,
        #                                   weight_decay=5e-4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        return F.log_softmax(h, dim=1)


class GraphSAGE(torch.nn.Module):
    """GraphSAGE"""

    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.sage1 = SAGEConv(dim_in, dim_h)
        self.sage2 = SAGEConv(dim_h, dim_out)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.sage1(x, edge_index).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage2(h, edge_index)
        return F.log_softmax(h, dim=1)


class GIN(torch.nn.Module):
    """GIN"""

    def __init__(self, dim_in, dim_h, dim_out):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(dim_in, dim_h),
                BatchNorm1d(dim_h),
                ReLU(),
                Linear(dim_h, dim_h),
                ReLU(),
            )
        )
        self.conv2 = GINConv(
            Sequential(
                Linear(dim_h, dim_h),
                BatchNorm1d(dim_h),
                ReLU(),
                Linear(dim_h, dim_h),
                ReLU(),
            )
        )
        self.conv3 = GINConv(
            Sequential(
                Linear(dim_h, dim_h),
                BatchNorm1d(dim_h),
                ReLU(),
                Linear(dim_h, dim_h),
                ReLU(),
            )
        )
        self.lin1 = Linear(dim_h * 3, dim_h * 3)
        self.lin2 = Linear(dim_h * 3, dim_out)

    def forward(self, data):
        # Node embeddings
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # # Graph-level readout
        # h1 = global_add_pool(h1, batch)
        # h2 = global_add_pool(h2, batch)
        # h3 = global_add_pool(h3, batch)

        # # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return F.log_softmax(h, dim=1)
