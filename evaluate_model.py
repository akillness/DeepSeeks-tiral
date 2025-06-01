import argparse
import os
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from unsloth import FastLanguageModel # Unsloth 모델 로딩에 필요할 수 있음
import ollama # Ollama 라이브러리 추가

def load_model_and_tokenizer(model_path, base_model_name=None, use_unsloth=False, max_seq_length=2048, dtype_str="bfloat16", load_in_4bit=False):
    """ 모델과 토크나이저를 로드합니다. LoRA 어댑터 또는 전체 모델을 로드할 수 있습니다. """
    print(f"Loading model from: {model_path}")
    dtype = getattr(torch, dtype_str)

    if use_unsloth:
        # Unsloth로 파인튜닝된 모델 또는 LoRA 어댑터 로드
        # base_model_name이 제공되면 LoRA 어댑터를 해당 베이스 모델에 적용하여 로드 시도
        model_to_load = base_model_name if base_model_name and os.path.exists(os.path.join(model_path, "adapter_config.json")) else model_path
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_to_load, # Base model or path to full Unsloth model
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        if base_model_name and model_to_load == base_model_name and os.path.exists(os.path.join(model_path, "adapter_config.json__")):
            print(f"Loading LoRA adapter from {model_path} onto base model {base_model_name}")
            try:
                # PEFT 모델에 어댑터를 로드하려고 시도
                if hasattr(model, 'load_adapter'):
                    model.load_adapter(model_path) # 저장된 어댑터 경로
                    print("Successfully loaded LoRA adapter via model.load_adapter().")
                else:
                    # FastLanguageModel.from_pretrained가 이미 어댑터를 처리했을 수 있음 (base_model_name이 아닌 model_path가 adapter 경로일 때)
                    # 또는 PEFT모델이 아닌 경우일 수 있으므로, 추가 로드 없이 진행
                    print("LoRA adapter might have been loaded by from_pretrained, or PEFT model not detected for explicit adapter loading.")
            except Exception as e:
                print(f"Could not load LoRA adapter: {e}. Ensure adapter files are present in {model_path}")
                print("Attempting to load as a full model or base model only.")
    else:
        # 일반 Hugging Face Transformers 모델 로드 (병합된 모델 또는 LoRA 미사용 모델)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            # load_in_4bit=load_in_4bit, # PEFT 없이 직접 4bit 로드 시 (필요시 설정)
        )
    
    if hasattr(model, 'to') and hasattr(model, 'device') and torch.cuda.is_available():
        model.to("cuda")
    if hasattr(model, 'eval'):
        model.eval() # 평가 모드
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, stream=True):
    """ 주어진 프롬프트로 텍스트를 생성합니다. """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    print(f"\n--- Prompt ---\n{prompt}")
    print("\n--- Generated Text (Transformers) ---")

    streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None

    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id, # 일반적이지만, 모델에 따라 다를 수 있음
            streamer=streamer,
        )
    
    if not stream:
        output_text = tokenizer.decode(generation_output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(output_text)
        return output_text
    return ""

def measure_throughput(model, tokenizer, prompt, num_tokens_to_generate=100, repetitions=5):
    """ 추론 처리량을 측정합니다 (tokens/sec). """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    total_time = 0
    total_tokens = 0

    print(f"\n--- Measuring Throughput (generating {num_tokens_to_generate} tokens, {repetitions} reps) ---")
    # Warm-up (첫 실행은 느릴 수 있음)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=num_tokens_to_generate, pad_token_id=tokenizer.eos_token_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(repetitions):
        start_time = time.time()
        with torch.no_grad():
            # 실제 생성된 토큰 수를 정확히 세기 위해 output을 받아야 함
            output = model.generate(
                **inputs,
                max_new_tokens=num_tokens_to_generate, 
                pad_token_id=tokenizer.eos_token_id,
                # do_sample=False, # 정확한 토큰 수 제어를 위해 샘플링 비활성화
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        generated_tokens = output.shape[1] - inputs.input_ids.shape[1]
        total_tokens += generated_tokens
        total_time += (end_time - start_time)
        print(f"Rep: Generated {generated_tokens} tokens in {end_time - start_time:.3f}s")

    avg_time_per_repetition = total_time / repetitions
    avg_tokens_per_repetition = total_tokens / repetitions
    tokens_per_second = avg_tokens_per_repetition / avg_time_per_repetition if avg_time_per_repetition > 0 else 0
    
    print(f"Average tokens generated per run: {avg_tokens_per_repetition:.2f}")
    print(f"Average time per run: {avg_time_per_repetition:.3f} s")
    print(f"Tokens per second: {tokens_per_second:.2f} tokens/s")
    return tokens_per_second

def interact_with_ollama(ollama_model_name, prompt, stream=True):
    """ Ollama API를 사용하여 지정된 모델과 상호작용합니다. """
    print(f"\n--- Interacting with Ollama model: {ollama_model_name} ---")
    print(f"--- Prompt ---\n{prompt}")
    print("\n--- Generated Text (Ollama) ---")
    try:
        response_stream = ollama.generate(
            model=ollama_model_name,
            prompt=prompt,
            stream=stream
        )
        full_response = ""
        if stream:
            for chunk in response_stream:
                if 'response' in chunk:
                    print(chunk['response'], end='', flush=True)
                    full_response += chunk['response']
                if 'error' in chunk:
                    print(f"\nError from Ollama: {chunk['error']}")
                    return None # 오류 발생 시 None 반환
            print() # 스트리밍 후 줄바꿈
        else:
            full_response = response_stream['response'] # stream=False 일때 전체 응답
            print(full_response)
        return full_response
    except Exception as e:
        print(f"\nError interacting with Ollama: {e}")
        print("Ensure Ollama is running and the model is available (e.g., 'ollama list').")
        print("You might need to pull the model first: 'ollama pull <model_name>' or create it from a GGUF file.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned language model or interact with Ollama.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the fine-tuned model (LoRA adapters checkpoint or merged model directory). Not used if --ollama_model is set.")
    parser.add_argument("--base_model_name", type=str, default=None, help="Base model ID if loading LoRA adapters (e.g., 'unsloth/llama-3-8b-Instruct-bnb-4bit'). Required if model_path contains only LoRA adapters and use_unsloth is True.")
    parser.add_argument("--use_unsloth_loader", action="store_true", help="Use Unsloth's FastLanguageModel for loading (recommended if model was trained with Unsloth).")
    
    # 모델 로딩 옵션 (load_model_and_tokenizer와 일치)
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for model loading.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Data type for model loading.")
    parser.add_argument("--load_in_4bit_eval", action="store_true", help="Load model in 4-bit for evaluation (if supported).")

    # 생성 옵션
    parser.add_argument("--prompt", type=str, default="What is the capital of France? Explain why it became the capital. ", help="Prompt for text generation. Use DeepSeek format with  and  tokens.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum new tokens to generate for qualitative test.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling. DeepSeek-R1 recommends 0.6 to reduce repetition and incoherence.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top_p for sampling. DeepSeek-R1 recommends 0.95.")
    parser.add_argument("--no_stream", action="store_true", help="Disable streaming output for qualitative test.")

    # 처리량 측정 옵션
    parser.add_argument("--measure_perf", action="store_true", help="Measure inference throughput.")
    parser.add_argument("--perf_tokens_to_generate", type=int, default=100, help="Number of tokens to generate for each throughput measurement run.")
    parser.add_argument("--perf_repetitions", type=int, default=10, help="Number of repetitions for throughput measurement.")

    # Ollama 관련 인자
    parser.add_argument("--ollama_model", type=str, default=None, help="Name of the model to use with Ollama (e.g., 'my-finetuned-model:latest'). If set, local model loading is skipped.")
    parser.add_argument("--ollama_no_stream", action="store_true", help="Disable streaming output for Ollama interaction.")

    args = parser.parse_args()

    if args.ollama_model:
        interact_with_ollama(args.ollama_model, args.prompt, stream=not args.ollama_no_stream)
    elif args.model_path:
        if args.use_unsloth_loader and not args.base_model_name and os.path.exists(os.path.join(args.model_path, "adapter_config.json__")):
            print("Warning: --use_unsloth_loader is active and model_path seems to be LoRA adapters, but --base_model_name is not provided. Please provide the base model name for correct LoRA loading.")
            # 여기서 프로그램을 종료하거나, 사용자에게 base_model_name을 다시 묻는 로직을 추가할 수 있습니다.
            # exit(1) # 또는 기본값으로 시도

        model, tokenizer = load_model_and_tokenizer(
            args.model_path, 
            base_model_name=args.base_model_name, 
            use_unsloth=args.use_unsloth_loader,
            max_seq_length=args.max_seq_length,
            dtype_str=args.dtype,
            load_in_4bit=args.load_in_4bit_eval
        )

        # 질적 평가 (텍스트 생성)
        generate_text(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_p, stream=not args.no_stream)

        # 성능 평가 (처리량)
        if args.measure_perf:
            measure_throughput(model, tokenizer, args.prompt, args.perf_tokens_to_generate, args.perf_repetitions)
    else:
        print("Please specify either --ollama_model to use Ollama or --model_path to load a local Transformers model.")

    print("\nEvaluation script finished.") 