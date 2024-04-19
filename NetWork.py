import torch
from torch.functional import Tensor
import torch.nn as nn

""" This script defines the network.
"""

class ResNet(nn.Module):
    def __init__(self,
            resnet_version,
            resnet_size,
            num_classes,
            first_num_filters,
        ):
        """
        1. Define hyperparameters.
        Args:
            resnet_version: 1 or 2, If 2, use the bottleneck blocks.
            resnet_size: A positive integer (n).
            num_classes: A positive integer. Define the number of classes.
            first_num_filters: An integer. The number of filters to use for the
                first block layer of the model. This number is then doubled
                for each subsampling block layer.
        
        2. Classify a batch of input images.

        Architecture (first_num_filters = 16):
        layer_name      | start | stack1 | stack2 | stack3 | output      |
        output_map_size | 32x32 | 32X32  | 16x16  | 8x8    | 1x1         |
        #layers         | 1     | 2n/3n  | 2n/3n  | 2n/3n  | 1           |
        #filters        | 16    | 16(*4) | 32(*4) | 64(*4) | num_classes |

        n = #residual_blocks in each stack layer = self.resnet_size
        The standard_block has 2 layers each.
        The bottleneck_block has 3 layers each.
        
        Example of replacing:
        standard_block      conv3-16 + conv3-16
        bottleneck_block    conv1-16 + conv3-16 + conv1-64

        Args:
            inputs: A Tensor representing a batch of input images.
        
        Returns:
            A logits Tensor of shape [<batch_size>, self.num_classes].
        """
        super(ResNet, self).__init__()
        self.resnet_version = resnet_version
        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = first_num_filters

        # define conv1
        self.start_layer = nn.Conv2d(3, self.first_num_filters, kernel_size=3, stride=1, padding=1, bias=False)

        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first block unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection.
        # Choose the block type
        block_fn = bottleneck_block if self.resnet_version == 2 else standard_block

        # Setup stack layers
        self.stack_layers = nn.ModuleList()
        current_filters = self.first_num_filters
        for i in range(3):  # Assuming 3 stages
            next_filters = current_filters * 2 if i > 0 else current_filters
            strides = 1 if i == 0 else 2
            self.stack_layers.append(stack_layer(next_filters, block_fn, strides, self.resnet_size, current_filters))
            current_filters = next_filters * (4 if self.resnet_version == 2 else 1)

        self.output_layer = output_layer(current_filters, self.resnet_version, self.num_classes)
    
    def forward(self, inputs):
        outputs = self.start_layer(inputs)
        for layer in self.stack_layers:
            outputs = layer(outputs)
        outputs = self.output_layer(outputs)
        return outputs

#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_relu_layer, self).__init__()

        self.bn = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:

        out = self.bn(inputs)
        out = self.relu(out)
        return out


class standard_block(nn.Module):
    """ Creates a standard residual block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(standard_block, self).__init__()

        self.projection_shortcut = projection_shortcut
        if projection_shortcut is not None:
            # Adjust projection shortcut to correctly handle the input channels
            self.projection_shortcut = nn.Sequential(
                nn.Conv2d(first_num_filters, filters, kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(filters))

        self.conv1 = nn.Conv2d(first_num_filters, filters, kernel_size=3, stride=strides, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)


    def forward(self, inputs: Tensor) -> Tensor:

        shortcut = self.projection_shortcut(inputs) if self.projection_shortcut is not None else inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += shortcut
        out = self.relu(out) 
        return out

class bottleneck_block(nn.Module):
    """ Creates a bottleneck block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution. NOTE: filters_out will be 4xfilters.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(bottleneck_block, self).__init__()
        
        self.expansion = 4
        out_channels = filters * self.expansion
        self.projection_shortcut = projection_shortcut
        if projection_shortcut is not None:
            self.projection_shortcut = nn.Sequential(
                nn.Conv2d(first_num_filters, out_channels, kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.projection_shortcut = nn.Identity()
            
            self.conv1 = nn.Conv2d(first_num_filters, filters, kernel_size=1, stride=1, bias=False)
            self.bn1 = nn.BatchNorm2d(filters)
            self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(filters)
            self.conv3 = nn.Conv2d(filters, out_channels, kernel_size=1, stride=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
            # Correctly placed ReLU instantiation
            self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        # The projection shortcut should come after the first batch norm and ReLU
		# since it performs a 1x1 convolution.
        shortcut = self.projection_shortcut(inputs) if self.projection_shortcut is not None else inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.relu(out)
        return out

class stack_layer(nn.Module):
    """ Creates one stack of standard blocks or bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
			    convolution in a block.
		block_fn: 'standard_block' or 'bottleneck_block'.
		strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
        resnet_size: #residual_blocks in each stack layer
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, block_fn, strides, resnet_size, first_num_filters) -> None:
        super(stack_layer, self).__init__()
        filters_out = filters * 4 if block_fn is bottleneck_block else filters

        self.blocks = nn.ModuleList()
        # projection_shortcut = ?
        # Only the first block per stack_layer uses projection_shortcut and strides
        
        # Determine the output number of filters
        filters_out = filters * 4 if block_fn is bottleneck_block else filters
        
        # Define the projection shortcut only for the first block if necessary
        projection_shortcut = None
        if strides != 1 or first_num_filters != filters_out:
            projection_shortcut = nn.Sequential(
                nn.Conv2d(first_num_filters, filters_out, kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(filters_out)
            )
        
        # Add the first block with potential downsampling and projection shortcut
        self.blocks.append(block_fn(filters=filters, projection_shortcut=projection_shortcut, strides=strides, first_num_filters=first_num_filters))
        
        # For the remaining blocks, the input and output sizes are the same,
        # and there's no downsampling, so strides are set to 1 and no projection shortcut.
        for _ in range(1, resnet_size):
            self.blocks.append(block_fn(filters=filters, projection_shortcut=None, strides=1, first_num_filters=filters_out))

    
    def forward(self, inputs: Tensor) -> Tensor:
        out = inputs
        for block in self.blocks:
            out = block(out)
        return out

class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        resnet_version: 1 or 2, If 2, use the bottleneck blocks.
        num_classes: A positive integer. Define the number of classes.
    """
    def __init__(self, filters, resnet_version, num_classes) -> None:
        super(output_layer, self).__init__()
        # Only apply the BN and ReLU for model that does pre_activation in each
		# bottleneck block, e.g. resnet V2.
        self.resnet_version = resnet_version
        if (resnet_version == 2):
            self.bn_relu = batch_norm_relu_layer(filters, eps=1e-5, momentum=0.997)

        
        # Apply global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer to produce class logits
        self.fc = nn.Linear(filters, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        # Conditional application of BN and ReLU based on resnet version
        if (self.resnet_version == 2):
            inputs = self.bn_relu(inputs)
        
        # Global average pooling
        inputs = self.global_avg_pool(inputs)
        inputs = torch.flatten(inputs, 1)
        # Fully connected layer for classification
        outputs = self.fc(inputs)
        
        return outputs
