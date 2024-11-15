import argparse
import os 
import torch
import pytorch_lightning as pl
from dataset import TextDataset, CharTokenizer
from cfg import get_config
from gpt import GPT

class GPTLightningModule(pl.LightningModule):

    def __init__(self, config, model, dataset):
        super().__init__()

        self.config = config
        self.model = model
        self.dataset = dataset
        #self.train_dataset = train_dataset
        print("running on device", self.device)
    
    def forward(self, x):
        # Forward pass through the model
        return self.model(x)

@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    model_type: str,
    prompt: str = "",
    num_samples: int = 10,
    n_steps: int = 20,
    do_sample: bool = True,
    top_k: int = None,
    top_p: float = 0.6,
    temperature: float = 1.0,
    device: str = "cpu",
    verbose: bool = True,
):
    """Generates text samples using a trained GPT model. This function takes a trained model and generates a specified number
    of text samples based on a given prompt. It allows for customization of the generation process through various parameters like the number
    of samples, the number of steps (tokens) to generate, sampling strategy, and others.

    Attributes:
        model (torch.nn.Module): The trained GPT model used for text generation.
        model_type (str): The type of GPT model used, necessary for the tokenizer.
        prompt (str, optional): The initial text prompt to start the generation. Defaults to an empty string for unconditional generation.
        num_samples (int, optional): The number of text samples to generate. Defaults to 10.
        n_steps (int, optional): The number of tokens to generate for each sample. Defaults to 20.
        do_sample (bool, optional): Whether to use sampling; set to False for deterministic generation. Defaults to True.
        top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to None.
        top_p (float, optional): The cumulative probability threshold for nucleus sampling. Defaults to 0.6.
        temperature (float, optional): The value used to module the next token probabilities. Defaults to 1.0.
        device (str, optional): The device (e.g., 'cpu' or 'cuda') on which to perform the computation. Defaults to 'cpu'.
        verbose (bool, optional): If True, prints each generated sample. Defaults to True.

    Notes:
        - The function uses the char level tokenizer we used for training.
        - The function is designed to handle both conditional and unconditional text generation based on the provided prompt.
    """
    dix = model.dataset.tokenizer.encode(prompt)
    # return as tensors
    x = torch.tensor(dix, dtype=torch.long).to(device).unsqueeze(0)
    
    
    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.model.generate(x, max_new_tokens=n_steps, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
    
    # Decode the predicted outputs 
    decoded_outputs = []
    for i in range(num_samples):
        #out = tokenizer.decode(y[i].cpu().squeeze())
        #print(y)
        out = ''.join([model.dataset.tokenizer.decode([int(k)]) for k in y[i].cpu().squeeze()])
        decoded_outputs.append(out)
        if verbose:
            print('-'*80)
            print(out)



if __name__ == "__main__":

    args = get_config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weights_folder', type=str, default='./logs/gpt-mini/version_0/checkpoints')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--num_generated_tokens', type=int, default=77)
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--prompt', type=str, default='Yesterday I went to the ')
    gen_args = parser.parse_args()
    for key, value in vars(gen_args).items():
        setattr(args, key, value)
    
    pl.seed_everything(args.seed) 

    # Load model weights    
    model_weights_path = os.path.join(args.model_weights_folder, sorted(os.listdir(args.model_weights_folder))[-1])
    state_dict = torch.load(model_weights_path)

    # Clean up state dict keys by removing '_orig_mod' prefix if present due to torch.compile()
    if state_dict['hyper_parameters']['compile'] and 'state_dict' in state_dict:
        cleaned_state_dict = {}
        for key, value in state_dict['state_dict'].items():
            new_key = key.replace('model._orig_mod.', 'model.')
            cleaned_state_dict[new_key] = value
        state_dict['state_dict'] = cleaned_state_dict
    
    # Initialize model
    default_cfg = GPT.get_default_config()
    saved_cfg = state_dict['hyper_parameters'] 
    
    saved_cfg = argparse.Namespace(**saved_cfg)

    # Convert Namespace objects to dictionaries before combining
    default_cfg_dict = vars(default_cfg)
    saved_cfg_dict = vars(saved_cfg)
    combined_cfg = {**default_cfg_dict, **saved_cfg_dict}
    
    # Create Namespace object from combined dictionary
    cfg = argparse.Namespace(**combined_cfg)
    gpt_model = GPT(cfg)

    # Setup dataset and model
    dataset = TextDataset(args, args.txt_file, args.block_size, CharTokenizer)
    model = GPTLightningModule(cfg, gpt_model, dataset)
    model.load_state_dict(state_dict['state_dict'])

    device = next(model.parameters()).device

    generate(
        prompt=args.prompt,
        model=model,
        model_type=args.model_type,
        num_samples=args.num_samples,
        n_steps=args.num_generated_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        device=device,
    )