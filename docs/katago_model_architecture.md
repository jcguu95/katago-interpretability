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

The shape of this `trunkfinal` tensor is `(N, C, H, W)`, where:

-   `N`: Batch size (the number of board positions being processed).
-   `C`: Number of channels in the trunk (`c_trunk`), a key hyperparameter of the model. For the model used in our tests (`kata1-b28c512nbt`), this value is 512.
-   `H`: Board height (e.g., 19).
-   `W`: Board width (e.g., 19).

This tensor is a rich, high-dimensional representation of the board state, which is then used by the final layers to make predictions.

### 4. The Policy and Value Heads (The Network's "Branches")

The `trunkfinal` tensor is the last *shared* representation. From this point, the network branches into separate "heads" that perform distinct tasks.

-   **Policy Head**: This head predicts the best move to play. The `trunkfinal` tensor is passed into `self.policy_head` (line 1967). It is important to note that in the source code, this tensor is named `out` when it is passed to the policy and value heads, but it is the same tensor that was just captured as `"trunkfinal"`. The `PolicyHead` module (defined at line 1421) uses its own small set of layers to produce policy logits for every possible move. The process is as follows:

    1.  **Input**: The `trunkfinal` tensor (passed as the `out` variable in the `Model.forward` method) serves as the direct input `x` to the `PolicyHead.forward` method (line 1475).

    2.  **Dual Branches**: The head immediately splits the computation into two parallel branches that both operate on the same `trunkfinal` input:
        -   **Spatial Branch ("p" branch)**: The tensor is passed through a 1x1 convolution (`self.conv1p`). This branch is responsible for generating location-specific move predictions on the board.
        -   **Global Branch ("g" branch)**: The tensor is also passed through a *separate* 1x1 convolution (`self.conv1g`). The result is then globally pooled (`self.gpool`) to summarize the board state into a single feature vector. This vector is then processed by several linear layers (`self.linear_g`, `self.linear_pass`, etc.) to compute features related to passing and other non-spatial aspects of the policy.

    3.  **Combination**: The output of the global branch (`outg`) is reshaped back into a spatial tensor (with dimensions `N, C, 1, 1`) and added to the output of the spatial branch (`outp`). This injects the global understanding of the position into the spatial predictions at every board location.

    4.  **Final Layers**: This combined tensor goes through a final normalization (`self.bias2`), activation (`self.act2`), and a final 1x1 convolution (`self.conv2p`) to produce the final logits for every possible move on the board. The logits for passing (calculated in the "g" branch) are then concatenated to these spatial logits to form the complete policy output.

-   **Value Head**: This head predicts the outcome of the game (win/loss probability, expected score, territory ownership, etc.). The *same* `trunkfinal` tensor is also passed into `self.value_head` (line 1975). The `ValueHead` module (defined at line 1531) has its own layers to produce these varied predictions.

---

### 5. Using `trunkfinal` for SAE Training

The core of our interpretability work involves training a Sparse Autoencoder (SAE) on the `trunkfinal` activations. It is important to understand how this high-dimensional tensor is processed for this purpose.

The `trunkfinal` tensor has a shape of `(C, H, W)` for a single board position, which is `(512, 19, 19)` for our model. Instead of treating this as a single, large `512 * 19 * 19 = 184,832`-dimensional vector, we make a simplifying architectural choice:

**We treat each of the `19 * 19 = 361` spatial locations as an independent data sample.**

For each sample, the feature vector is the set of `512` channel activations at that specific `(x, y)` board coordinate.

This means we reformat the `(N, C, H, W)` batch of activations into a tensor of shape `(N * H * W, C)`, or `(N * 361, 512)`. The SAE is then trained on these `512`-dimensional vectors.

This approach is based on the hypothesis that the channel features have a consistent meaning regardless of their spatial location, and it makes the problem of training an SAE far more tractable.

#### Limitations of This Approach

As noted, this approach discards all spatial relationships between different locations on the board. Go concepts like "a group of stones" or "a critical zone" are inherently spatial. The current SAE model cannot learn features that represent these multi-location concepts directly. Instead, it can only learn to identify features present at a single point, based on the 512-dimensional channel vector at that point.

This is a deliberate simplification. The benefit is a much smaller, more tractable model. The limitation is that we may miss out on features that are defined by the spatial arrangement of activations across the board. Future work could explore more complex, spatially-aware architectures like convolutional autoencoders to capture these relationships.
