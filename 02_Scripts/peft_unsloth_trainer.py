import torch
import wandb
from transformers import TrainingArguments
from datasets import load_dataset
from enum import Enum
from unsloth import FastLanguageModel, standardize_sharegpt, is_bfloat16_supported
from trl import SFTTrainer

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ModelType(Enum):
    # 모델이 추가되면 Enum에 적절하게 추가
    LGAI_EXAONE3_5_8B_INSTRUCT = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    META_LLAMA3_1_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"
    UNSLOTH_LLAMA3_1_8B_INSTRUCT = "unsloth/Llama-3.1-8B-Instruct"
    GEMMA = "gemma"
    ALPACA = "Alpaca"

def get_model_type(model_id: str):
    """model_id를 기반으로 ModelType을 반환"""
    for model_type in ModelType:
        if model_type.value in model_id.lower():
            return model_type
    raise ValueError("지원되지 않는 모델입니다. 모델명을 확인하세요.")

def get_model_id(model_type: ModelType):
    """ModelType을 기반으로 model_id를 반환"""
    return model_type.value

def utrain_sft(model_type = ModelType.META_LLAMA3_1_8B_INSTRUCT, max_seq_length = 1024,
            wandb_project = 'fintuning', wandb_key="", 
            train_data_path="./your data path.json", test_data_path="./your data path.json",
            lorar = 8, loraa = 16, loradropout = 0.05, 
            epochs= 2, batch_size = 4, gradient_step = 2, learning_rate = 1e-4):

    # 모델 ID 설정
    model_id = get_model_id(model_type) 
    print({model_id})
    # 모델 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id, 
        max_seq_length = max_seq_length,
        dtype = None,# None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True, # False for LoRA 16bit
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
    )
    print(f"[info] {model_id} 로드완료")

    # 데이터셋 로드
    train_dataset, test_dataset = format_dataset(
        model_type,
        train_data_files=train_data_path,
        test_data_files=test_data_path
    )
    train_dataset = standardize_sharegpt(train_dataset['train'])
    test_dataset = standardize_sharegpt(test_dataset['train'])

    # LoRA 설정
    model = FastLanguageModel.get_peft_model(
        model,
        r = lorar, 
        lora_alpha = loraa,  
        lora_dropout = loradropout,  # 0이면 최적화된 성능을 보장. 작은 값으로 설정하면 과적합 방지 가능.
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  
        bias = "none",  # "none"이 최적화된 값. 필요에 따라 "all" 또는 "lora_only"로 변경 가능.

        # Unsloth의 최적화된 VRAM 사용 방식 적용 (최대 30% 절약, 2배 큰 배치 크기 가능)
        use_gradient_checkpointing = "unsloth",  # "unsloth" 또는 True 사용 가능. 메모리 절약 효과.
        random_state = 3407,  # 실험의 일관성을 유지하기 위한 랜덤 시드 값.
        # Rank Stabilized LoRA (RSLoRA) 사용 여부
        use_rslora = False,  # False면 일반 LoRA 사용. True면 RSLoRA 적용 가능 (추가적인 안정성 제공).
        loftq_config = None,  # None이면 사용 안 함. 사용하면 LoftQ 기반 저비트 양자화 적용 가능.
    )

    # 저장 폴더 설정
    outName = f"unsloth-{model_id.split('/')[-1]}-{epochs}-{batch_size}-{gradient_step}-{learning_rate}-{lorar}-{loraa}-{loradropout}"
    output_dir = f"./99_GitLoss/01_RoLaModels/{outName}"

    # 학습 설정
    
    trainer = SFTTrainer(
    model = model,  # 학습할 모델 (LoRA 적용된 LLaMA 등)
    tokenizer = tokenizer,  # 토크나이저 (문장을 토큰 단위로 변환)

    train_dataset = train_dataset,  # 학습에 사용할 데이터셋
    eval_dataset=test_dataset,

    dataset_text_field = "text",  # 데이터셋에서 텍스트 필드 이름 지정

    max_seq_length = max_seq_length,  # 최대 시퀀스 길이 설정
    dataset_num_proc = 1,  # 데이터 전처리 멀티프로세싱 개수 설정 (속도 최적화)

    packing = False,  # ✅ `True`로 설정하면 짧은 시퀀스를 묶어 학습 속도를 5배 증가 가능!

    # 학습 파라미터 설정
    args = TrainingArguments(
        evaluation_strategy="steps", 
        eval_steps=2, 

        per_device_train_batch_size = batch_size,  
        gradient_accumulation_steps = gradient_step, 

        num_train_epochs=epochs, 

        optim = "adamw_torch",

        learning_rate = learning_rate, 
        lr_scheduler_type = "linear", 

        fp16 = not is_bfloat16_supported(),  # GPU 환경에 따라 FP16 사용 여부 자동 선택
        bf16 = is_bfloat16_supported(),  # Ampere 이상 GPU에서는 BF16 사용  (A100, RTX 30/40)은 BF16 사용
    
        weight_decay = 0.01,  # 가중치 감소 (Regularization)

        warmup_ratio=0.1, # 상승곡선
        # warmup_steps = 5,  # 초기 워밍업 스텝 설정

        seed = 3407,  # 랜덤 시드 설정 (재현성 보장)

        report_to = "wandb" if wandb_key else 'none',

        output_dir=output_dir,
        save_steps = 10,
        save_strategy="steps", #epoch,steps

        logging_steps = 2,
        log_level="info" # debug, info, warning, error
        ),
    )

    # wandb 설정
    if wandb_key:
        wandb.finish()
        wandb.login(key=wandb_key)
        wandb.init(
            project= wandb_project,
            name=outName,
            reinit=True,
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_step,
                "num_train_epochs": epochs
            }
        )
    print(f"[info] 학습 시작")
    # 학습 시작

    trainer_stats = trainer.train()

    # 모델 저장
    saveLoRA_dir = f"{output_dir}/lora_model"
    model.save_pretrained(saveLoRA_dir) # Local saving
    tokenizer.save_pretrained(saveLoRA_dir)
    print(f"[info] 로라 어뎁터 저장 {saveLoRA_dir}")

    return model, tokenizer, output_dir

def umerge_model_load(lora_path: str):
    """
    LoRA 모델을 로드하고 원래 모델과 병합.
    """
    merged_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = lora_path, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 1024,
        dtype = None,# None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True, # False for LoRA 16bit
    )
    FastLanguageModel.for_inference(merged_model) # Enable native 2x faster inference
    return merged_model, tokenizer

def umodel_save(model, tokenizer, save_path: str, save_method = "merged_16bit"):
    """
    save_method = merged_16bit, merged_4bit, lora
    """
    
    model.save_pretrained_merged(f"{save_path}/merged", tokenizer, save_method = "merged_16bit",)

def uchat_response(model, tokenizer, user_input , max_tokens=1024, do_sample=False):
    """
    범용적으로 사용할 수 있는 챗봇 응답 생성 함수.
    
    Args:
        model: Pre-trained LLM 모델
        tokenizer: 모델에 맞는 토크나이저
        user_input (str): 사용자의 질문
        system_prompt (str): 시스템 역할 프롬프트
        max_tokens (int): 최대 생성할 토큰 수
        do_sample (bool): 샘플링 사용 여부 (False이면 결정론적 응답)
    Returns:
        str: 모델이 생성한 응답
    """
    system_prompt = (
    "You are a reliable and trustworthy AI assistant. Please follow these guidelines:\n\n"
    "1. Always provide accurate and verified information based on facts. Do not generate false or speculative content under any circumstances.\n"
    "2. If you are unsure or lack sufficient information, respond with 'I don't know' or indicate that more context is needed.\n"
    "3. Always maintain a polite, respectful, and professional tone regardless of the user's behavior. Keep your language neutral and unbiased.\n"
    "4. Remember the context of previous interactions and maintain consistency throughout multi-turn conversations.\n"
    "5. Ensure your responses are logically structured and well-organized. Provide detailed explanations when necessary, but avoid unnecessary repetition.\n"
    "6. Do not include biased, offensive, or speculative statements. Avoid expressing personal opinions unless explicitly requested.\n"
    "7. Adapt your tone and style based on the user’s intent, level of formality, and the nature of the question."
    )

    # 메시지 구성
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # 채팅 템플릿 적용하여 토크나이징
    source = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # 모델 응답 생성
    with torch.no_grad():
        outputs = model.generate(
            source.to("cuda"),
            max_new_tokens=max_tokens,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
            do_sample=do_sample
        )

    # 출력 변환 및 정리
    response = tokenizer.decode(outputs[0][source.shape[-1]:], skip_special_tokens=True)
    response = response.replace('\n', ' ').strip()

    return response

# 모델별 Chat Template
chat_templates = {
    #EXAONE, LLAMA, GEMMA, ALPACA 외 다른 템플릿이 필요하면 추가
    "EXAONE": """[|system|]You are EXAONE model from LG AI Research, a helpful assistant.[|endofturn|]

[|user|]{INPUT}

[|assistant|]{OUTPUT}[|endofturn|]""",

    "LLAMA": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
아래는 작업을 설명하는 지시사항입니다. 입력된 내용을 바탕으로 적절한 응답을 작성하세요.<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>""",

    "GEMMA": """<start_of_turn>system
### SYSTEM:
아래는 작업을 설명하는 지시사항입니다. 입력된 내용을 바탕으로 적절한 응답을 작성하세요.
<end_of_turn>
<start_of_turn>user
### INSTRUCTION:
{INPUT}
<end_of_turn>
<start_of_turn>model
### RESPONSE:
{OUTPUT}
<end_of_turn>""",

    "ALPACA": """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{INPUT}

### Response:
{OUTPUT}""",
}
def format_dataset(model_type: ModelType, train_data_files : str, test_data_files=None):
    """
    Returns:
        tuple: 변환된 (train_dataset, test_dataset) - test_dataset은 없을 경우 None 반환
    """

    for key in chat_templates:
        if key in model_type.name:
            matched_template = chat_templates[key]
            break
    
    if matched_template is None:
        raise ValueError(f"지원되지 않는 모델명입니다: {model_type}")

    # 모델별 템플릿 선택
    def formatting_prompts_func(examples):
        """각 데이터 샘플에 템플릿 적용"""
        return {
            "text": matched_template.format(
                INPUT=examples["instruction"].strip(),
                OUTPUT=examples["output"].strip()
            )
        }

    # 학습 데이터 로드
    train_dataset = load_dataset("json", data_files=train_data_files)
    train_dataset = train_dataset.map(formatting_prompts_func, remove_columns=['instruction', 'output'])

    # 테스트 데이터 로드 (있을 경우만)
    test_dataset = None
    if test_data_files:
        test_dataset = load_dataset("json", data_files=test_data_files)
        test_dataset = test_dataset.map(formatting_prompts_func, remove_columns=['instruction', 'output'])

    return train_dataset, test_dataset