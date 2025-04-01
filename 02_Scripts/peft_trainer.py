import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, Gemma3ForCausalLM
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
from enum import Enum
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ModelType(Enum):
    # 모델이 추가되면 Enum에 적절하게 추가
    LGAI_EXAONE3_5_8B_INSTRUCT = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    META_LLAMA3_1_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"
    UNSLOTH_LLAMA3_1_8B_INSTRUCT = "unsloth/Llama-3.1-8B-Instruct"
    GOOGLE_GEMMA3_12B_IT = "google/gemma-3-12b-it"
    ALPACA = "Alpaca"

class LoraTarget(Enum):
    SMALL = ["q_proj", "v_proj"]
    NORMAL = ["q_proj", "o_proj", "k_proj", "v_proj"]
    FULL = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

def get_model_type(model_id: str):
    """model_id를 기반으로 ModelType을 반환"""
    for model_type in ModelType:
        if model_type.value in model_id.lower():
            return model_type
    raise ValueError("지원되지 않는 모델입니다. 모델명을 확인하세요.")

def get_model_id(model_type: ModelType):
    """ModelType을 기반으로 model_id를 반환"""
    return model_type.value

def train_sft(model_type : ModelType, use_quantization = True, max_seq_length = 1024,
            wandb_project = 'fintuning', wandb_key="", 
            train_data_path="./00_Data/train_data_v6.json", test_data_path="./00_Data/eval_data_v6.json",
            lorar = 8, loraa = 16, loradropout = 0.05, targetmodule = LoraTarget.FULL,
            epochs= 2, batch_size = 4, gradient_step = 2, learning_rate = 1e-4):

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 모델 ID 설정
    model_id = get_model_id(model_type) 

    # 모델 로드
    model, tokenizer = model_load( model_type, use_quantization)

    if use_quantization:
        model = prepare_model_for_kbit_training(model) # 양자화 모델을 훈련할 수 있도록
    
    model.gradient_checkpointing_enable()#훈련 시 메모리 절약 (출력값을 필요할 때만 계산)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length= max_seq_length,
        use_fast=True,
        )
    print(f"[info] {model_id} 로드완료")

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
        target_modules= targetmodule.value,
        bias="none",
    )

    # 저장 폴더 설정
    outName = f"{model_id.split('/')[-1]}-{epochs}-{batch_size}-{gradient_step}-{learning_rate}-{lorar}-{loraa}-{loradropout}-{targetmodule}"
    output_dir = f"./99_GitLoss/01_RoLaModels/{outName}"

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

        fp16=True,
        bf16=False,

        weight_decay = 0.01,

        warmup_ratio=0.1,
        # warmup_steps = 5

        seed = 3407,

        report_to = "wandb" if wandb_key else 'none',

        output_dir=output_dir,
        save_steps = 10,
        save_strategy="steps", #epoch,steps

        logging_steps = 2,
        log_level="info" # debug, info, warning, error
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
    print(f"[info] 학습 시작")
    # 학습 시작

    model.config.use_cache = False
    trainer.train()

    # 모델 저장
    saveLoRA_dir = f"{output_dir}/lora_model"
    trainer.model.save_pretrained(saveLoRA_dir)
    print(f"[info] 로라 어뎁터 저장 {saveLoRA_dir}")

    model.eval() # 모델의 가중치는 변경하지 않고, forward 연산만 수행함.
    model.config.use_cache = True  # 이전 계산 결과를 저장하고 사용	추론 속도 빨라짐, 메모리 사용 증가
    
    return model, tokenizer, output_dir

def model_load(model_type:ModelType, use_quantization:bool):
    """
    모델 로드.
    """
    # 모델 ID 설정정
    model_id = get_model_id(model_type) 

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
        torch_dtype = torch.bfloat16   # 일반 로드 시 bfloat16,float16 사용
        quantization_config = None  # 양자화 없음

    # 모델 및 토크나이져 로드드
    if ModelType.GOOGLE_GEMMA3_12B_IT == model_type: 
        model = Gemma3ForCausalLM.from_pretrained(
        model_id, 
        device_map = "auto", 
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        )
    else :
        model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  # {"":0}, 자동으로 GPU 할당 
        torch_dtype=torch_dtype,  # 양자화 안 하면 bfloat16 적용
        quantization_config=quantization_config,  # 양자화 적용 여부
        trust_remote_code= True if 'exaone' in model_id.lower() else False,  # 원격 코드 신뢰 여부
        )
            
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer

def lora_model_load(model_type:ModelType, lora_path: str, use_quantization :bool):
    """
    LoRA 모델을 로드.
    """

    base_model, tokenizer = model_load(model_type, use_quantization)
    model  = PeftModel.from_pretrained(base_model, lora_path, device_map="auto")
    print(model.peft_config)  # LoRA 설정이 잘 들어왔는지
    return model, tokenizer

def merge_model_load(model_type:ModelType, lora_path: str, use_quantization : False):
    """
    LoRA 모델을 로드하고 원래 모델과 병합.
    """

    lora_model, tokenizer = lora_model_load(model_type, lora_path, use_quantization)
    merged_model  = lora_model.merge_and_unload() #실제 병합
    print(type(merged_model)) 

    return merged_model, tokenizer

def merge_model_load_save(model_type : ModelType, lora_path: str):
    """
    LoRA 모델을 로드하고 원래 모델과 병합한 후 저장하는 함수.
    """
    merged_model, tokenizer = merge_model_load(model_type, lora_path, False)

    save_path = f'{lora_path}/merged'
    model_save(merged_model, tokenizer, save_path)

    print(f"모델 병합 및 저장 완료: {save_path}")

    return merged_model, tokenizer

def model_save(model, tokenizer, save_path: str):
    """
    모델 저장.
    """
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

def chat_response(model, tokenizer, user_input, max_tokens=1024, temperature = 0.7, top_p = 0.95,  do_sample=False, system_prompt = ""):
    """
    범용적으로 사용할 수 있는 챗봇 응답 생성 함수.
    """
    default_system_prompt = (
        "당신은 사용자의 요청에만 충실하게 답변하는 사실 기반의 중립적인 AI 어시스턴트입니다.\n\n"

        "역할 (ROLE):\n"
        "- 당신은 일반 지식, 상식, 정보성 질문에 정확하게 답변하는 데 특화되어 있습니다.\n"

        "제한 사항 (RESTRICTIONS):\n"
        "- 사용자의 의도를 추측하거나 유추하지 말고, 입력된 문장에 명시적으로 나타난 단어에만 기반하여 판단하십시오.\n"
        "- 반드시 **한글로만** 답변하십시오. 영어를 포함한 다른 언어는 절대 사용하지 마십시오.\n\n"

        "사용자 의도 분류 (INTENT CLASSIFICATION):\n"
        "- 숨겨진 의도나 배경을 추측하지 말고, 명확히 표현된 단어와 문장에만 기반하여 응답하십시오.\n\n"

        "행동 원칙 (BEHAVIOR):\n"
        "- 답변할 수 없는 경우에는 '모르겠습니다.' 또는 '추가 설명이 필요합니다.'라고 응답하십시오.\n"
        "- 응답은 간결하고, 사실에 기반하며, 질문에 직접적으로 관련된 내용만 포함해야 합니다.\n"
        "- 항상 정중하고, 중립적이며, 정보를 전달하는 어조를 유지하십시오.\n"
    )


    if not system_prompt:
        system_prompt = default_system_prompt

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
            source.to(model.device),
            max_new_tokens=max_tokens,
            eos_token_id=tokenizer.eos_token_id,
            temperature= temperature,
            top_p= top_p,
            do_sample=do_sample #True 여야 다양성 반영됨
        )

    # 출력 변환 및 정리
    response = tokenizer.decode(outputs[0][source.shape[-1]:], skip_special_tokens=True)
    response = response.replace('\n', ' ').strip()

    return response

# 모델별 Chat Template
chat_templates = {
    #EXAONE, LLAMA, GEMMA, ALPACA 외 다른 템플릿이 필요하면 추가
    #You are EXAONE model from LG AI Research, a helpful assistant.
    "EXAONE": """[|system|]You are EXAONE model from LG AI Research, a helpful assistant.[|endofturn|]

[|user|]
[Category: {CATEGORY}]
{INPUT}
[|assistant|]{OUTPUT}[|endofturn|]""",

    "LLAMA": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.<|eot_id|><|start_header_id|>user<|end_header_id|>

[Category: {CATEGORY}]
{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>""",

    "GEMMA": """<bos><start_of_turn>system
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
<end_of_turn>
<start_of_turn>user
[Category: {CATEGORY}]
{INPUT}<end_of_turn>
<start_of_turn>model
{OUTPUT}
<end_of_turn>""",

    "ALPACA": """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
[Category: {CATEGORY}]
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
                OUTPUT=examples["output"].strip(),
                CATEGORY=(examples.get("category") or "").strip()
            )
        }

    # 학습 데이터 로드
    train_dataset = load_dataset("json", data_files=train_data_files)
    train_dataset = train_dataset.map(formatting_prompts_func, remove_columns=['instruction', 'output'])
    print(f"[TrainDataLen] {len(train_dataset['train'])}")
    # 테스트 데이터 로드 (있을 경우만)
    test_dataset = None
    if test_data_files:
        test_dataset = load_dataset("json", data_files=test_data_files)
        test_dataset = test_dataset.map(formatting_prompts_func, remove_columns=['instruction', 'output'])
        print(f"[TestDataLen] {len(test_dataset['train'])}")

    return train_dataset, test_dataset





