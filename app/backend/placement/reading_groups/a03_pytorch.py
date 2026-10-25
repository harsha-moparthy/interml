from . import G

T = "https://pytorch.org/tutorials"
D = "https://pytorch.org/docs/stable"

TENSORS = ("Tensors (PyTorch tutorial)", f"{T}/beginner/basics/tensorqs_tutorial.html", "PyTorch", "Creating tensors, shape and dtype, and operations such as reshape and transpose.")
AUTOGRAD = ("Automatic differentiation with torch.autograd", f"{T}/beginner/basics/autogradqs_tutorial.html", "PyTorch", "Shows requires_grad, backward and the computational graph with a small example.")
BUILD = ("Build the Neural Network", f"{T}/beginner/basics/buildmodel_tutorial.html", "PyTorch", "Defines nn.Module subclasses and nn.Sequential layer by layer.")
NNTUT = ("What is torch.nn really?", f"{T}/beginner/nn_tutorial.html", "PyTorch", "Rebuilds a training loop from raw tensors up to nn.Module, optim and DataLoader.")
DATA = ("Datasets & DataLoaders", f"{T}/beginner/basics/data_tutorial.html", "PyTorch", "Custom Dataset classes and batching with DataLoader.")
SAVE = ("Saving and Loading Models", f"{T}/beginner/saving_loading_models.html", "PyTorch", "state_dict, checkpoints and loading weights safely.")
AMP = ("Automatic Mixed Precision examples", f"{D}/notes/amp_examples.html", "PyTorch docs", "autocast and GradScaler patterns including gradient accumulation and clipping.")

GROUPS = [
    G((40, 41), "tensor reshape dtype", TENSORS, NNTUT),
    G(42, "autograd backward requires_grad", AUTOGRAD, ("Autograd mechanics", f"{D}/notes/autograd.html", "PyTorch docs", "How the graph is recorded and how backward walks it.")),
    G((43, 44, 53), "module optimizer loss", BUILD, NNTUT),
    G(45, "state_dict load", SAVE, ("Saving and Loading Models", f"{T}/beginner/basics/saveloadrun_tutorial.html", "PyTorch", "Short walk-through of saving and loading weights for a trained model.")),
    G((46, 48), "module parameter", BUILD, ("ModuleList", f"{D}/generated/torch.nn.ModuleList.html", "PyTorch docs", "Why a list of layers must be a ModuleList to be registered.")),
    G(47, "hook forward module", ("Forward hooks in PyTorch", "https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/", "Nandita Bhaskhar", "Shows how to register forward hooks and capture intermediate activations."), ("nn.Module", f"{D}/generated/torch.nn.Module.html", "PyTorch docs", "register_forward_hook and the other hook methods.")),
    G(49, "transfer learning freeze parameters", ("Transfer Learning for Computer Vision Tutorial", f"{T}/beginner/transfer_learning_tutorial.html", "PyTorch", "Freezes backbone layers and trains a new head."), ("Optimizer", f"{D}/optim.html", "PyTorch docs", "Parameter groups with per-group learning rates.")),
    G((50, 51, 52), "dataset dataloader batch", DATA, ("torch.utils.data", f"{D}/data.html", "PyTorch docs", "Custom collate_fn, samplers including WeightedRandomSampler, and batching.")),
    G(54, "autograd function backward", ("Extending PyTorch: custom autograd Functions", f"{D}/notes/extending.html", "PyTorch docs", "Subclass torch.autograd.Function with forward and backward and save tensors for backward."), ("PyTorch: Defining new autograd Functions", f"{T}/beginner/examples_autograd/two_layer_net_custom_function.html", "PyTorch", "A runnable example of a custom ReLU Function.")),
    G(55, "autocast mixed precision", AMP, ("Automatic Mixed Precision recipe", f"{T}/recipes/recipes/amp_recipe.html", "PyTorch", "Introduces autocast and GradScaler step by step.")),
    G(57, "checkpoint memory activation", ("Activation Checkpointing Techniques", "https://pytorch.org/blog/activation-checkpointing-techniques/", "PyTorch blog", "Explains recomputing activations in backward to save memory and the variants."), ("torch.utils.checkpoint", f"{D}/checkpoint.html", "PyTorch docs", "API for checkpointing a function or sequential block.")),
]
