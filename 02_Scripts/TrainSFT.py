import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
from enum import Enum
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ModelType(Enum):
    EXAONE = "EXAONE"
    LLAMA = "LLaMA"
    GEMMA = "gemma"
    ALPACA = "Alpaca"

def trainSFT(model_id ="meta-llama/Llama-3.1-8B-Instruct", use_quantization = True,
            max_seq_length = 1024, wandb_key="", train_data_path="",
            test_data_path="",  lorar = 8, loraa = 16, loradropout = 0.05, 
            epochs= 2, batch_size = 4, gradient_step = 2, learning_rate = 1e-4):

    # 모델 타입 설정
    if 'exaone' in model_id.lower():
        model_type = ModelType.EXAONE
    elif 'llama' in model_id.lower():
        model_type = ModelType.LLAMA
    elif 'gemma' in model_id.lower():
        model_type = ModelType.GEMMA
    elif 'alpaca' in model_id.lower():
        model_type = ModelType.ALPACA
    else:
        print("모델명을 확인해 주세요.")

    #  양자화 설정 (사용할 경우)
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4-bit 양자화 활성화
        bnb_4bit_use_double_quant=True,  # 이중 양자화 적용
        bnb_4bit_quant_type="nf4",  # nf4 양자화 방식 사용
        bnb_4bit_compute_dtype=torch.bfloat16,  # 연산 시 bfloat16 사용
        )
        torch_dtype = None  # 양자화 시에는 torch_dtype 사용하지 않음
        quantization_config = bnb_config  # 양자화 적용
    else:
        torch_dtype = torch.bfloat16  # 일반 로드 시 bfloat16 사용
        quantization_config = None  # 양자화 없음

    # 모델 및 토크나이져 로드드
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  # 자동으로 GPU 할당
        torch_dtype=torch_dtype,  # 양자화 안 하면 bfloat16 적용
        quantization_config=quantization_config,  # 양자화 적용 여부
        trust_remote_code= True if ModelType.EXAONE else False,  # 원격 코드 신뢰 여부
      )
    if use_quantization:
        model = prepare_model_for_kbit_training(model) # 양자화 모델을 훈련할 수 있도록
    
    model.gradient_checkpointing_enable()#훈련 시 메모리 절약 (출력값을 필요할 때만 계산)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length= max_seq_length,
        use_fast=True,
        )
    print(f"{model_id} Loaded")

    # 토크나이저 패딩 설정
    if 'llama' in model_id.lower():
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 128004
        tokenizer.padding_side = 'right'
    else:
        tokenizer.padding_side = 'right'

    # 데이터셋 로드
    train_dataset, test_dataset = format_dataset(
        model_type,
        train_data_files=train_data_path,
        test_data_files=test_data_path
    )

    # LoRA 설정
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lorar,
        lora_alpha=loraa,
        lora_dropout=loradropout,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    print(f"Loraconfig Loaded")

    # 저장 폴더 설정
    wandb_project = f"finetuning"
    outName = f"{model_id.split('/')[-1]}-{epochs}-{batch_size}-{gradient_step}-{learning_rate}-{lorar}-{loraa}-{loradropout}"
    output_dir = f"../01_Models/01_RoLaModels/{outName}"

    # 학습 설정
    train_args = TrainingArguments(
        # per_device_eval_batch_size = 2,
        # eval_accumulation_steps = 4,
        evaluation_strategy = "steps",
        eval_steps = 2,

        gradient_checkpointing=True,
      
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_step,

        num_train_epochs=epochs,

        optim = "adamw_torch",

        learning_rate=learning_rate,
        lr_scheduler_type="linear",

        fp16=False,
        bf16=True,

        weight_decay = 0.01,

        warmup_ratio=0.1,
        # warmup_steps = 5

        seed = 3407,

        report_to = "wandb" if wandb_key else 'none',
        logging_steps = 2,
        output_dir=output_dir,
        save_steps = 50,

        save_strategy="epoch", #epoch,steps
        log_level="debug"
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=lora_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset['train'],
        eval_dataset=test_dataset['train'],
        args=train_args,
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

    print('train start')
    # 학습 시작

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model.config.use_cache = False
    trainer.train()

    # 모델 저장
    saveLoRA_dir = f"{output_dir}/LoRA"
    trainer.model.save_pretrained(saveLoRA_dir)
    print(f"Model saved at {saveLoRA_dir}")


    model.eval() # 모델의 가중치는 변경하지 않고, forward 연산만 수행함.
    model.config.use_cache = True  # 이전 계산 결과를 저장하고 사용	추론 속도 빨라짐, 메모리 사용 증가
    return model, tokenizer


def merge_lora_model(model_id, lora_path):
    """
    LoRA 모델을 로드하고 원래 모델과 병합.
    """
    model_id = model_id
    peft_model_id = lora_path

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # float16로 유지 bfloat16
        device_map="cuda"
    )

    merged_model  = PeftModel.from_pretrained(base_model, peft_model_id, device_map="cuda")
    merged_model  = merged_model.merge_and_unload() #실제 병합

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return merged_model, tokenizer

def merge_lora_model_save(model_id, lora_path, save_path):
    """
    LoRA 모델을 로드하고 원래 모델과 병합한 후 저장하는 함수.
    """
    merged_model, tokenizer = merge_lora_model(model_id, lora_path)

    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"모델 병합 및 저장 완료: {save_path}")

    return merged_model, tokenizer

def chat_response(model, tokenizer, user_input, system_prompt="You are a bot that responds to weather queries." , max_tokens=1024, do_sample=False):
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
    ModelType.EXAONE: """[|system|]You are EXAONE model from LG AI Research, a helpful assistant.[|endofturn|]

[|user|]{INPUT}

[|assistant|]{OUTPUT}[|endofturn|]""",

    ModelType.LLAMA: """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
아래는 작업을 설명하는 지시사항입니다. 입력된 내용을 바탕으로 적절한 응답을 작성하세요.<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>""",

    ModelType.GEMMA: """<start_of_turn>system
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

    ModelType.ALPACA: """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{INPUT}

### Response:
{OUTPUT}""",
}

def format_dataset(model_name: ModelType, train_data_files, test_data_files=None):
    """
    Returns:
        tuple: 변환된 (train_dataset, test_dataset) - test_dataset은 없을 경우 None 반환
    """
    if model_name not in chat_templates:
        raise ValueError(f"지원되지 않는 모델명입니다: {model_name}")

    # 모델별 템플릿 선택
    chat_template = chat_templates[model_name]

    def formatting_prompts_func(examples):
        """각 데이터 샘플에 템플릿 적용"""
        return {
            "text": chat_template.format(
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