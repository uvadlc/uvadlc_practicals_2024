import argparse

import torch
import torchmetrics
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

import pytorch_lightning as pl
from  pytorch_lightning.loggers  import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder


from gpt import GPT
from dataset import TextDataset, CharTokenizer
from generate import generate as generate_pretrained
from cfg import get_config


class GPTLightningModule(pl.LightningModule):

    def __init__(self, config, model, train_dataset):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        if config.compile:
            model = torch.compile(model)
        self.model = model

        self.train_dataset = train_dataset
        print("running on device", self.device)

        # Unpack config hparams
        # NOTE: LearningRateFinder callback needs access to a self.lr
        self.lr = self.config.learning_rate
    
    def forward(self, x):
        # Forward pass through the model
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        # Calculate loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        self.log('train_loss', loss, on_epoch=True)

        # Sample from predictions to calc accuracy
        acc = self.calc_accuracy_from_logits(logits, y)
        self.log('train_acc', acc, on_epoch=True)

        # Generate some sentences once in a while
        if self.global_step % self.config.generate_every_n_steps == 0:
            generated_sents = self.generate()
            self.logger.experiment.add_text('Training texts', generated_sents, self.global_step)
        return loss
    

    def calc_accuracy_from_logits(self, logits: torch.Tensor, targets: torch.Tensor):
        """ Calculates the accuracy of predictions from logits against the true targets. This function processes 
        a batch of logits (predictions before applying softmax) to calculate the accuracy of the model's predictions.
        It applies top-k filtering to the logits, computes the softmax probabilities, and then determines the top 
        predictions. The accuracy is computed by comparing these predictions with the true target values.

        Parameters:
            - logits (torch.Tensor): A tensor of logits from the model. Shape is typically (batch_size, sequence_length, vocab_size).
            - targets (torch.Tensor): The true target values. Shape is typically (batch_size, sequence_length).

        Returns:
            - torch.Tensor: The calculated accuracy as a tensor.
        """
        idx = torch.empty((targets.shape[1], 1)).to(targets.device)
        with torch.no_grad():
            for token_logits in logits:
                v, _ = torch.topk(token_logits, 50)
                token_logits[token_logits < v[:, [-1]]] = -float('Inf')
                token_probs = F.softmax(token_logits, dim=-1)
                _, idx_next = torch.topk(token_probs, k=1, dim=-1)
                idx = torch.cat((idx, idx_next), dim=1)
        idx = idx[:,1:]
        acc = torchmetrics.functional.accuracy(idx.T, targets, task='multiclass', num_classes=self.config.vocab_size)
        return acc
        

    @torch.inference_mode()
    def generate(self, prompt: str = '', num_samples: int = 5, n_steps: int = 30, do_sample: bool = True, top_k: int = None, top_p: float=0.6, verbose: bool = False):
        """ Generates text based on a given prompt using either a pre-trained model or a custom-trained model. This function 
        generates text by conditioning on an input prompt. It supports both pre-trained and custom-trained models. For pre-trained models,
        it delegates to a `generate_pretrained` function. For custom-trained models, it starts from a default context or the provided prompt 
        and generates text using the model's `generate` method. For the pretrained model we use the seperate function, since we then need to 
        rely on a pretrained tokenizer of Huggingface.

        Parameters:
            - prompt (str, optional): The initial text prompt for text generation. Defaults to an empty string, which triggers a default context for custom models.
            - num_samples (int, optional): The number of text samples to generate. Only used with pre-trained models. Defaults to 5.
            - n_steps (int, optional): The number of tokens to generate. Defaults to 30.
            - do_sample (bool, optional): Whether to use sampling for text generation. Defaults to True.
            - top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to None.
            - top_p (float, optional): The cumulative probability threshold for nucleus sampling. Defaults to 0.6.
            - verbose (bool, optional): If True, enables verbose output. Currently not used in the function body. Defaults to False.

        Returns:
            - str or list of str: The generated text. If using a pre-trained model, a list of generated samples is returned. For a custom model, 
                                  a single string of generated text is returned.
        """

        if self.config.use_pretrained:
            decoded_outputs = generate_pretrained(
                prompt=prompt, 
                num_samples=num_samples,
                steps=n_steps,
                do_sample=do_sample,
                device=self.config.device,
                verbose=verbose,
            )
        else:
            context = 'Yesterday I went ' if prompt == '' else prompt
            x = torch.tensor(self.train_dataset.tokenizer.encode(context), dtype=torch.long)[None,...].to(self.config.device)
            y = self.model.generate(x, n_steps, temperature=1.0, do_sample=do_sample, top_k=top_k, top_p=top_p)[0]
            decoded_outputs = self.train_dataset.tokenizer.decode(y.tolist())
        return decoded_outputs


    def configure_optimizers(self):
        # Function to pass the optimizer to pytorch-lightning
        optimizer = self.model.configure_optimizers(self.config)
        return optimizer
    
    
    def train_dataloader(self):
        # Setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size, 
            sampler=RandomSampler(self.train_dataset, replacement=True),
            shuffle=False,
            drop_last=True, 
            pin_memory=True,
            num_workers=self.config.num_workers,
        )
        return train_loader

def train(args):
    """
    Function for training and testing a GPT model.
    Inputs:
        args (Namespace) - Namespace object from the argument parser
    """
    print(args)
    pl.seed_everything(args.seed)  
    
    if args.pretrained_tokenizer:
        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")
        args.vocab_size = tokenizer.max_token_value
    else:
        tokenizer = CharTokenizer(args.txt_file)
        args.vocab_size = tokenizer.vocab_size  # Set vocab size from tokenizer
    # Create the dataset with the tokenizer
    dataset = TextDataset(args, args.txt_file, args.block_size, tokenizer)

    # Initialise the gpt-model
    if args.use_pretrained:
        gpt_model = GPT.from_pretrained(model_type=args.model_type)
    else:
        cfg = GPT.get_default_config()
        cfg.model_type = args.model_type
        cfg.block_size = args.block_size
        cfg.vocab_size = args.vocab_size
        cfg.use_flash_attn = args.use_flash_attn
        cfg.compile = args.compile
        cfg.abs_emb = args.abs_emb
        gpt_model = GPT(config=cfg)

    # Assuming `model` and `train_dataset` are defined and `config` is your configuration object
    lightning_model = GPTLightningModule(args, gpt_model, dataset)

    # Setup logger
    logger = TensorBoardLogger(args.log_dir, name=args.model_type)

    # Create generate callback
    save_callback = ModelCheckpoint(save_weights_only=True, mode="min", monitor="train_loss")
    lr_callback = LearningRateFinder()

    if "16" in args.precision:
        # Reduce even further precision in computations that might use it.
        torch.set_float32_matmul_precision("medium")

    # Initialize a pytorch-lightning trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[save_callback, lr_callback],
        max_epochs=args.num_epochs,
        accelerator=args.device,
        enable_progress_bar=args.progress_bar,
        gradient_clip_val=args.clip_grad_norm,
        precision=args.precision
    )

    # Train the model
    trainer.fit(lightning_model)


if __name__ == "__main__":
    args = get_config()
    args.device = ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")) 

    train(args=args)