import argparse
import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

def main(args):
    # 1. 모델 로드 및 Unsloth 설정
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=getattr(torch, args.dtype), # 예: torch.bfloat16
        load_in_4bit=args.load_in_4bit, # 4bit 양자화 사용
        # token = "hf_...", # Access private repositories
    )

    # LoRA 설정 (Unsloth에서 자동으로 최적의 r, alpha 등을 찾아줄 수 있음)
    # 또는 명시적으로 지정
    if args.use_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r, # rank
            target_modules=args.lora_target_modules if args.lora_target_modules else \
                           ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 일반적인 모듈
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=42,
            max_seq_length=args.max_seq_length,
        )
        print("LoRA adapters configured.")

    # 2. 데이터셋 준비
    # 이 부분은 사용자의 데이터셋 형식에 맞게 수정 필요
    # 예시: Alpaca 스타일 데이터셋 (text 컬럼 하나에 프롬프트와 응답이 포함된 경우)
    # dataset = load_dataset("json", data_files={"train": args.dataset_path_train, "eval": args.dataset_path_eval})
    # train_dataset = dataset["train"]
    # eval_dataset = dataset["eval"]

    # 예시: 간단한 텍스트 파일에서 데이터셋 로드 (각 줄이 학습 샘플)
    # 실제 사용 시에는 instruction, input, output 등을 포함하는 구조화된 데이터셋 권장
    def formatting_prompts_func(examples):
        # 이 함수는 데이터셋의 각 샘플을 모델 입력 형식에 맞게 변환합니다.
        # DeepSeek-R1 모델은 " " 와 " " 토큰을 사용합니다.
        # 
        # 예상 데이터셋 형식 (JSON/JSONL):
        # 1. {"text": "전체 대화 형식이 이미 포함된 텍스트"}
        # 2. {"instruction": "질문", "input": "추가 입력 (optional)", "output": "답변"}
        # 3. {"user": "사용자 메시지", "assistant": "어시스턴트 응답"}
        #
        # 사용자의 데이터셋에 맞게 아래 로직을 수정해야 합니다.
        texts = []
        for i in range(len(examples.get(list(examples.keys())[0], []))):
            # 방법 1: 이미 완성된 대화 형식이 'text' 컬럼에 있는 경우
            if "text" in examples and examples["text"][i].strip():
                text = examples["text"][i]
                # 이미 DeepSeek 형식인지 확인, 아니면 추가
                if " " not in text and " " not in text:
                    text = f" {text} "
                texts.append(text)
            
            # 방법 2: instruction, input, output 형식인 경우 (Alpaca 스타일)
            elif "instruction" in examples and "output" in examples:
                instruction = examples["instruction"][i]
                input_text = examples.get("input", [""] * len(examples["instruction"]))[i]
                output = examples["output"][i]
                
                if input_text.strip():
                    prompt = f" {instruction}\n\n{input_text} {output}"
                else:
                    prompt = f" {instruction} {output}"
                texts.append(prompt)
            
            # 방법 3: user, assistant 형식인 경우
            elif "user" in examples and "assistant" in examples:
                user_msg = examples["user"][i]
                assistant_msg = examples["assistant"][i]
                prompt = f" {user_msg} {assistant_msg}"
                texts.append(prompt)
            
            # 기본값: 첫 번째 컬럼을 사용
            else:
                first_key = list(examples.keys())[0]
                text = examples[first_key][i]
                if not text.startswith(" "):
                    text = f" {text} "
                texts.append(text)
        
        return {"text": texts}
    
    print(f"Loading dataset from: {args.dataset_path_train}")
    dataset = load_dataset(args.dataset_format, data_files={"train": args.dataset_path_train})
    train_dataset = dataset["train"].map(formatting_prompts_func, batched = True,)
    
    eval_dataset = None
    if args.dataset_path_eval:
        print(f"Loading evaluation dataset from: {args.dataset_path_eval}")
        eval_dataset_raw = load_dataset(args.dataset_format, data_files={"eval": args.dataset_path_eval})
        eval_dataset = eval_dataset_raw["eval"].map(formatting_prompts_func, batched = True,)


    # 3. 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        # max_steps=args.max_steps, # num_train_epochs 대신 사용 가능
        learning_rate=args.learning_rate,
        fp16=not args.load_in_4bit, # 4bit 아니면 fp16 사용 (bf16은 dtype으로 제어)
        bf16=args.dtype == "bfloat16" and not args.load_in_4bit,
        logging_steps=args.logging_steps,
        optim=args.optimizer,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=42,
        save_strategy="epoch" if eval_dataset else "no", # 평가셋 있을때만 epoch마다 저장
        evaluation_strategy="epoch" if eval_dataset else "no",
        load_best_model_at_end=True if eval_dataset else False,
        # ... 기타 SFTTrainer, TrainingArguments 옵션들
    )

    # 4. Trainer 초기화 및 학습 시작
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",  # formatting_prompts_func 에서 "text"로 반환
        max_seq_length=args.max_seq_length,
        args=training_args,
        # packing=True, # 시퀀스를 묶어 효율성 향상 (데이터셋에 따라 필요)
    )

    print("Starting training...")
    trainer.train()

    # 5. 모델 저장
    final_model_path = os.path.join(args.output_dir, "final_model")
    print(f"Saving final model to {final_model_path}")
    if args.use_lora:
        # LoRA 어댑터만 저장
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print("LoRA adapters saved.")

        if args.merge_lora_after_training:
            # (선택 사항) LoRA 어댑터를 기본 모델과 병합하여 저장
            # 병합된 모델은 더 쉽게 배포할 수 있지만, 추가 학습은 어려움
            print("Merging LoRA adapters with the base model...")
            merged_model_path = os.path.join(args.output_dir, "final_merged_model")
            # Unsloth는 from_pretrained로 다시 로드하여 merge_and_unload()를 지원하거나,
            # PEFT 라이브러리의 merge_and_unload() 직접 사용 가능
            try:
                # 다시 로드하여 병합 (Unsloth 방식)
                merged_model, merged_tokenizer = FastLanguageModel.from_pretrained(
                    model_name = args.model_name, # Base model
                    max_seq_length = args.max_seq_length,
                    dtype = getattr(torch, args.dtype),
                    load_in_4bit = args.load_in_4bit, # Usually False for merging, or merge in higher precision
                )
                merged_model.load_adapter(final_model_path) # Load trained adapters
                merged_model.merge_and_unload() # Merge
                merged_model.save_pretrained(merged_model_path)
                merged_tokenizer.save_pretrained(merged_model_path)
                print(f"Merged model saved to {merged_model_path}")
            except Exception as e:
                print(f"Error during LoRA merge: {e}. Merged model not saved.")
    else:
        # 전체 모델 저장 (LoRA 미사용 시)
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print("Full fine-tuned model saved.")

    print("Fine-tuning completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model using Unsloth and QLoRA.")

    # 모델 인자
    parser.add_argument("--model_name", type=str, default="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit", help="Hugging Face model ID. For DeepSeek fine-tuning, consider: unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit or unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Data type for the model (e.g., bfloat16).")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load model in 4-bit for QLoRA.")
    parser.add_argument("--no_load_in_4bit", action="store_false", dest="load_in_4bit", help="Do not load model in 4-bit.")

    # LoRA 인자
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA for fine-tuning.")
    parser.add_argument("--no_use_lora", action="store_false", dest="use_lora", help="Do not use LoRA (full fine-tuning).")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--lora_target_modules", nargs='+', default=None, help="LoRA target modules (e.g., q_proj v_proj). If None, uses Unsloth defaults.")
    parser.add_argument("--merge_lora_after_training", action="store_true", help="Merge LoRA adapters with the base model after training and save it.")


    # 데이터셋 인자
    parser.add_argument("--dataset_path_train", type=str, required=True, help="Path to the training data file or directory.")
    parser.add_argument("--dataset_path_eval", type=str, default=None, help="Path to the evaluation data file or directory.")
    parser.add_argument("--dataset_format", type=str, default="json", help="Format of the dataset (e.g., json, csv, text). Passed to load_dataset.")
    # `formatting_prompts_func`를 데이터셋 형식에 맞게 수정해야 함.

    # 학습 인자 (TrainingArguments 일부)
    parser.add_argument("--output_dir", type=str, default="./output_finetune", help="Output directory for checkpoints and final model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device train batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.") # Unsloth은 더 높은 LR을 권장하기도 함
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps.")
    parser.add_argument("--optimizer", type=str, default="adamw_8bit", help="Optimizer to use (e.g., adamw_torch, adamw_8bit).") # Unsloth은 paged_adamw_8bit 등도 지원
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="LR scheduler type.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps.")
    # parser.add_argument("--max_steps", type=int, default=-1, help="If > 0, total number of training steps to perform.")

    args = parser.parse_args()

    if not args.load_in_4bit and args.dtype == "bfloat16":
        print("Warning: load_in_4bit is False, but dtype is bfloat16. Ensure your GPU supports bfloat16 for full precision training.")
    if args.load_in_4bit and not args.use_lora:
        print("Warning: Model loaded in 4-bit but LoRA is not enabled. Full fine-tuning in 4-bit is generally not recommended or supported.")
        print("Consider enabling LoRA with --use_lora or disabling 4-bit loading with --no_load_in_4bit for full fine-tuning.")

    main(args) 