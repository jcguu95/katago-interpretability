# KataGo's Neural Network Architecture

This document provides a high-level overview of KataGo's neural network architecture, based on an analysis of the PyTorch model implementation in `katago/python/katago/train/model_pytorch.py`.

The network uses a standard and powerful "residual network" (ResNet) design. It can be understood as having three main parts: an **Input Processing Stage**, a deep **Residual Trunk**, and separate **Output Heads**.

The data flow is: **Input -> Trunk -> `trunkfinal` -> {Policy Head, Value Head}**.

Our feature extraction process targets the `'trunkfinal'` layer. This is the activation from the second-to-last stage of the network, which represents the model's final, rich, internal representation of the game state before it is used to compute specific predictions.

---

### 1. Input Processing (The Network's "Roots")

The model begins by processing two different kinds of input and combining them into a single tensor. This happens at the start of the `Model.forward` method (around line 1893).

-   **Spatial Features**: This is the main board data (stone positions, liberties, history, etc.), represented as a tensor of shape `(N, C, H, W)`. It is processed by a single convolutional layer (`self.conv_spatial`, defined at line 1749).
-   **Global Features**: This is non-spatial data about the game state (komi, whose turn it is, rules, etc.), represented as a flat vector. It is processed by a single linear layer (`self.linear_global`, defined at line 1752).

These two processed inputs are added together (line 1903: `out = x_spatial + x_global`) to form the initial tensor for the main network body.

### 2. The Residual Trunk (The Network's "Body")

The bulk of the network is the "trunk," a deep stack of residual blocks. This is where the model performs most of its computation, progressively refining its understanding of the position.

-   In the `Model.__init__` method, a `torch.nn.ModuleList` named `self.blocks` is created (line 1769). The model's configuration file specifies the number and type of blocks (e.g., `ResBlock`, `BottleneckResBlock`).
-   In the `Model.forward` method, the input tensor is passed sequentially through every block in this list (the loop at line 1952). Each block processes the output of the previous one and adds its own input back via a "residual connection." This technique is crucial for successfully training very deep networks.

### 3. The `trunkfinal` Layer (Top of the Trunk)

After the tensor has passed through all residual blocks, it undergoes one final transformation before being sent to the output layers. This is precisely where the `trunkfinal` activation is generated.

Within the `Model.forward` method, after the loop over `self.blocks` is complete, the following steps occur:

1.  `out = self.norm_trunkfinal(out, ...)` (line 1960): A final normalization layer is applied.
2.  `out = self.act_trunkfinal(out)` (line 1961): A final activation function (e.g., ReLU) is applied.
3.  **`extra_outputs.report("trunkfinal", out)`** (line 1964): The resulting tensor is explicitly captured and given the name `"trunkfinal"`.

### 4. The Policy and Value Heads (The Network's "Branches")

The `trunkfinal` tensor is the last *shared* representation. From this point, the network branches into separate "heads" that perform distinct tasks.

-   **Policy Head**: This head predicts the best move to play. The `trunkfinal` tensor is passed into `self.policy_head` (line 1967). The `PolicyHead` module (defined at line 1421) uses its own small set of layers to produce policy logits for every possible move.
-   **Value Head**: This head predicts the outcome of the game (win/loss probability, expected score, territory ownership, etc.). The *same* `trunkfinal` tensor is also passed into `self.value_head` (line 1975). The `ValueHead` module (defined at line 1531) has its own layers to produce these varied predictions.
