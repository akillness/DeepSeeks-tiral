import os
import argparse
from huggingface_hub import snapshot_download, HfFolder
from requests.exceptions import ConnectionError
from huggingface_hub.utils import HfHubHTTPError

def download_model_from_hf(repo_id: str, local_dir: str, allow_patterns: list[str] | None = None, token: str | None = None):
    """
    Downloads a model or specific files from Hugging Face Hub.

    Args:
        repo_id (str): The repository ID on Hugging Face (e.g., "unsloth/DeepSeek-R1-0528-GGUF").
        local_dir (str): The local directory to save the model files.
        allow_patterns (list[str], optional): A list of glob patterns to filter files.
                                             If None, downloads all files in the repository.
                                             Example: ["*.gguf", "tokenizer.model"]
        token (str, optional): Hugging Face API token for private repositories.
                               If None, attempts to use saved token or anonymous download.
    """
    print(f"Attempting to download from repository: {repo_id}")
    print(f"Saving to local directory: {local_dir}")
    if allow_patterns:
        print(f"Applying file patterns: {allow_patterns}")
    else:
        print("No specific file patterns provided, will attempt to download all matching files.")

    # HF_HUB_ENABLE_HF_TRANSFER 환경 변수 설정 (0으로 설정 시 rate limit 방지 가능성)
    # hf_transfer는 대용량 파일 전송에 유용하지만, 때때로 문제가 발생할 수 있음
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "1") # 기본값 1 (활성화)
    print(f"HF_HUB_ENABLE_HF_TRANSFER is set to: {os.environ['HF_HUB_ENABLE_HF_TRANSFER']}")

    try:
        if token:
            print("Using provided Hugging Face token.")
        elif HfFolder.get_token():
            print("Using Hugging Face token found in local cache.")
            token = HfFolder.get_token()
        else:
            print("No Hugging Face token provided or found in cache. Proceeding with anonymous download if possible.")

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            local_dir_use_symlinks=False, # Windows 호환성을 위해 False 권장
            token=token,
            # resume_download=True, # 중단된 다운로드 이어받기 (필요시 활성화)
            # etag_timeout=60, # ETag 타임아웃 증가 (대용량 파일 다운로드 시 유용)
        )
        print(f"Model files successfully downloaded to {os.path.abspath(local_dir)}")
        if allow_patterns:
            print(f"Downloaded files matching patterns: {allow_patterns}")

    except HfHubHTTPError as e:
        print(f"Hugging Face Hub HTTP Error: {e}")
        if e.response.status_code == 401:
            print("Authentication failed. If this is a private repository, please provide a valid token.")
        elif e.response.status_code == 404:
            print(f"Repository or files not found: {repo_id} with patterns {allow_patterns}")
        else:
            print("An HTTP error occurred. Please check your network connection and the repository URL.")
    except ConnectionError as e:
        print(f"Connection Error: {e}")
        print("Failed to connect to Hugging Face Hub. Please check your internet connection.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please ensure you have 'huggingface_hub' and 'hf_transfer' (if enabled) installed correctly.")
        print("If downloading a private model, ensure your token has the correct permissions.")
        print("You might also need to log in to Hugging Face CLI: 'huggingface-cli login'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from Hugging Face Hub.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF", # DeepSeek R1-0528 Qwen3-8B GGUF 모델로 변경
        help="Repository ID on Hugging Face (e.g., 'unsloth/DeepSeek-R1-0528-GGUF')."
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None, # 기본값을 None으로 변경하여 repo_id 기반으로 동적 설정
        help="Local directory to save the model files. If None, defaults to './cache/REPO_OWNER/REPO_NAME'."
    )
    parser.add_argument(
        "--allow_patterns",
        type=str,
        nargs="*", # 여러 패턴을 받을 수 있도록
        default=["*Q4_K_XL*"], # DeepSeek 블로그에서 권장하는 Q4_K_XL 양자화 버전
        help='''List of glob patterns to filter files (e.g., "*.gguf" "*.json"). '''
             '''If not provided, attempts to download all files. '''
             '''Example: --allow_patterns "*Q4_K_XL*" "*UD-Q2_K_XL*" "*.txt" '''
             '''DeepSeek-R1-0528 recommended quantizations: "*Q4_K_XL*", "*UD-Q2_K_XL*", "*UD-IQ1_S*" '''
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token for private repositories. If not provided, tries to use cached token."
    )
    parser.add_argument(
        "--disable_hf_transfer",
        action="store_true",
        help="Disable hf_transfer by setting HF_HUB_ENABLE_HF_TRANSFER to '0'."
    )

    args = parser.parse_args()

    if args.disable_hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        print("hf_transfer explicitly disabled via command line argument.")

    # local_dir 기본값 설정
    if args.local_dir is None:
        if "/" not in args.repo_id:
            # repo_id가 'gpt2'처럼 네임스페이스가 없는 경우
            args.local_dir = os.path.join(".", "cache", args.repo_id)
        else:
            # repo_id가 'unsloth/llama-3-8b'처럼 네임스페이스가 있는 경우
            repo_owner, repo_name = args.repo_id.split("/", 1)
            args.local_dir = os.path.join(".", "cache", repo_owner, repo_name)
        print(f"local_dir not specified, defaulting to: {args.local_dir}")

    # 로컬 디렉토리 생성 (존재하지 않는 경우)
    if not os.path.exists(args.local_dir):
        os.makedirs(args.local_dir)
        print(f"Created local directory: {args.local_dir}")

    download_model_from_hf(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        allow_patterns=args.allow_patterns if args.allow_patterns else None, # 빈 리스트 대신 None 전달
        token=args.token
    )

# 예시 사용법:
# 기본값으로 DeepSeek UD-Q2_K_XL 모델 다운로드:
# python download_model.py

# 다른 모델 (예: gpt2) 전체 다운로드:
# python download_model.py --repo_id gpt2 --local_dir models/gpt2 --allow_patterns

# 특정 패턴의 파일만 다운로드 (예: Q4_K_M GGUF 모델 및 설정 파일):
# python download_model.py --repo_id unsloth/DeepSeek-R1-0528-GGUF --local_dir models/my_deepseek_q4 --allow_patterns "*Q4_K_M*.gguf" "*.json" "tokenizer.model"

# 비공개 모델 다운로드 (토큰 사용):
# python download_model.py --repo_id your_username/private_model --local_dir models/private_model --token YOUR_HF_TOKEN

# hf_transfer 비활성화하고 다운로드:
# python download_model.py --disable_hf_transfer 