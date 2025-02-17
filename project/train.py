import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from datasets import load_dataset
from transformers import Trainer

# Configuration
class TrainingConfig:
    def __init__(self):
        self.student_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        self.teacher_model_id = "Qwen/Qwen2.5-72B-Instruct"
        self.output_dir = "./qwen_distill_output"
        self.num_train_epochs = 1
        self.per_device_train_batch_size = 4
        self.gradient_accumulation_steps = 4
        self.learning_rate = 2e-5
        self.kd_ratio = 0.5
        self.temperature = 2.0
        self.max_grad_norm = 0.5
        self.logging_steps = 100
        self.save_steps = 1000

config = TrainingConfig()

# QLoRA Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# LoRA Configuration
lora_config = LoraConfig(
    r=8,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM",
)

def setup_distributed_training():
    """Initialize distributed training environment"""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group("nccl")
            return local_rank
    return -1

# Dataset preparation
class FunctionCallingDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=1024):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Combine messages into a single text
        full_text = ""
        for message in item["conversations"]:
            role = message["from"]
            content = message["value"]
            full_text += f"{role}: {content}\n"
        
        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"][0],
            "attention_mask": encodings["attention_mask"][0],
            "labels": encodings["input_ids"][0].clone()
        }

# Distillation loss function
def distillation_loss_fn(student_logits, teacher_logits, temperature=2.0):
    """Compute the knowledge distillation loss"""
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(
        student_probs, 
        teacher_probs, 
        reduction='batchmean'
    ) * (temperature ** 2)

# Modified Trainer for distributed distillation
class DistributedDistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.accelerator = Accelerator()
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # Move inputs to the correct device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get student outputs
        student_outputs = model(**inputs)
        
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        # Compute distillation loss
        distillation_loss = distillation_loss_fn(
            student_outputs.logits, 
            teacher_outputs.logits,
            temperature=config.temperature
        )
        
        # Combine losses
        loss = student_outputs.loss + config.kd_ratio * distillation_loss
        
        return (loss, student_outputs) if return_outputs else loss

def load_models(local_rank):
    """Load and prepare the student and teacher models"""
    # Load student model with QLoRA
    student_model = AutoModelForCausalLM.from_pretrained(
        config.student_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    student_model = get_peft_model(student_model, lora_config)
    
    # Load teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # Print trainable parameters if main process
    if local_rank in [-1, 0]:
        student_model.print_trainable_parameters()
    
    return student_model, teacher_model

def prepare_training_args(local_rank):
    """Prepare training arguments"""
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=True,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        max_grad_norm=config.max_grad_norm,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        local_rank=local_rank
    )

def main():
    # Setup distributed training
    local_rank = setup_distributed_training()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.student_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load models
    student_model, teacher_model = load_models(local_rank)
    
    # Load and prepare dataset
    dataset = load_dataset("hypervariance/function-calling-sharegpt")
    train_dataset = FunctionCallingDataset(tokenizer, dataset["train"])
    
    # Prepare training arguments
    training_args = prepare_training_args(local_rank)
    
    # Initialize trainer
    trainer = DistributedDistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )
    
    # Start training
    trainer.train()
    
    # Save the final model (only on main process)
    if local_rank in [-1, 0]:
        trainer.save_model()
        
if __name__ == "__main__":
    main()