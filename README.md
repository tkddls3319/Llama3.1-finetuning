## 02_Scripts폴더의 SFT_meta__Llama-3.1-8B-Instruct.ipynb을 실행 시켜 사용해보세요

좀 더 자세한 설명은 https://usingsystem.tistory.com/560 에 있습니다.


# Llama 3.1 LoRA 기반 PEFT 적용 가이드

LoRA (Low-Rank Adaptation)는 LLM(대형 언어 모델)의 일부 파라미터만 미세 조정하여 메모리와 계산량을 절약하는 파인튜닝 기법입니다.

Llama 3.1 모델에 LoRA를 적용하여 PEFT(Parameter-Efficient Fine-Tuning)를 수행하는 방법을 이론과 실전 코드 예제를 통해 단계별로 설명합니다.

---

## 📌 PEFT(LoRA)란 무엇인가?

[Hugging Face PEFT 공식 문서](https://github.com/huggingface/peft)

### 🔹 PEFT (Parameter-Efficient Fine-Tuning)
기존 LLM을 풀 파인튜닝(Full Fine-Tuning)하지 않고, 적은 수의 파라미터만 학습하는 방법입니다. 대표적인 방식으로 LoRA, QLoRA, Adapter, Prompt Tuning 등이 있습니다.

#### 🔹 LoRA(Low-Rank Adaptation)
- 모델의 모든 가중치를 업데이트하지 않고, 일부 모듈만 미세 조정하는 기법입니다.
- **메모리 절약 + 빠른 학습 + 성능 유지**가 가능합니다.

#### 🔹 LoRA 적용 방식 (기존 LLM vs LoRA 적용 LLM 비교)

**[ 기존 LLM 모델 ]**
```
├── Query Layer
├── Key Layer
├── Value Layer
├── Fully Connected Layer (100% 업데이트)
```

**[ LoRA 적용 LLM 모델 ]**
```
├── Query Layer (LoRA Adapter 추가)
├── Key Layer (LoRA Adapter 추가)
├── Value Layer (LoRA Adapter 추가)
├── Fully Connected Layer (동결됨)
```

---

## 📌 Batch, Step, Epoch 개념

### 🏷️ Batch, Step, Epoch은 LLM 학습 과정에서 데이터를 처리하는 단위입니다.
```python
per_device_train_batch_size = 2  # 각 디바이스(GPU 또는 TPU)에서 사용할 학습 배치 크기
per_device_eval_batch_size = 2   # 각 디바이스(GPU 또는 TPU)에서 사용할 평가(검증) 배치 크기

logging_steps = 2  # 학습 중 로그를 출력할 간격 (즉, 2 스텝마다 로그 출력)
evaluation_strategy = 'steps'  # 모델 평가를 실행할 기준 ('steps'는 특정 스텝마다 평가 수행)
eval_steps = 100  # 평가(evaluation)를 수행할 스텝 간격 (100 스텝마다 검증 수행)
save_steps = 1000  # 모델을 저장할 간격 (1000 스텝마다 모델 체크포인트 저장)
num_train_epochs = 3  # 전체 학습 데이터셋을 3번 반복하여 학습 (즉, 에포크 수는 3)
```
---

## 1. Batch (배치)
Batch(배치)는 한 번의 학습 단계에서 사용되는 데이터 샘플의 개수를 의미합니다. 즉, 한 번의 Training Step에서 모델이 학습하는 데이터 묶음을 뜻합니다.

### Batch Size
- **Batch Size**: 한 번의 Step에서 모델이 처리하는 데이터 샘플 개수
- 배치 크기는 학습 속도, 메모리 사용량, 일반화 성능 등에 영향을 미침
- 메모리가 허용한다면 큰 배치 크기를 사용하는 것이 효율적

### 배치 크기의 영향
#### 작은 배치 크기
- 학습 과정이 불안정해질 수 있지만, 일반화 성능은 더 좋아지는 경향
- 메모리 사용량이 적어 제한된 자원으로도 학습 가능하지만, 전체 학습 시간이 길어질 수 있음

#### 큰 배치 크기
- 학습이 안정적이지만, 세부적인 특징을 놓쳐 과적합(Overfitting)의 위험이 있음
- 더 많은 메모리가 필요하지만 학습 시간이 단축될 수 있음
- 메모리가 허용된다면 큰 배치 크기를 사용하는 것이 효율적
- 다만, 큰 배치를 통해 과적합의 징후가 나타난다면 배치를 줄여 일반화 성능을 높이는 것도 방법

```python
per_device_train_batch_size = 8  # 배치 크기
per_device_eval_batch_size = 8
```

### 과적합(Overfitting)
과적합이란 모델이 학습 데이터에는 매우 잘 맞지만, 새로운 데이터에서는 성능이 저하되는 현상을 의미합니다.

예를 들어, 파인튜닝 과정에서 학습 데이터의 Loss는 지속적으로 감소하지만, 평가 데이터의 Loss는 오히려 증가하는 경우가 발생할 수 있습니다.

#### 과적합의 원인
- 지나치게 큰 배치 크기
- 과도한 반복 학습
- 작은 학습 데이터셋

특히, 학습 데이터셋이 작을 경우 배치 크기가 크면 데이터의 세부적인 특징을 놓칠 가능성이 높아집니다. 따라서 배치 크기를 늘릴 때는 충분한 학습 데이터가 확보되어 있는지를 고려하는 것이 중요합니다.

흥미로운 점은, 특정 도메인에 특화된 모델을 만들 때는 일부러 과적합을 유도하기도 한다는 것입니다. 이 경우, 특정 도메인 데이터의 특성을 더욱 정밀하게 학습하도록 유도하여 해당 분야에서 최적화된 모델을 구축하는 방식으로 접근합니다.

---

## 2. Step (학습 스텝)
Training Step이란 모델의 **파라미터가 한 번 업데이트되는 과정**을 의미합니다.
즉, Batch 크기만큼 데이터를 학습하고, 이를 기반으로 가중치(모델 파라미터)를 조정하는 과정입니다.

### 예제
- Batch 크기가 10일 경우, 한 번의 Training Step에서 10개의 샘플을 학습 후 모델 업데이트 수행

#### Step 계산 공식
```python
데이터셋 크기 = 10,000개
Batch 크기 = 10
Training Step 수 = 전체 데이터 개수 / Batch 크기
# → 10,000 / 10 = 1,000 Step
# → 즉, 1,000번의 Step이 진행되면 데이터셋 전체가 한 번 학습됨
```

---

## 3. Epoch (에포크)
Epoch이란 **학습 전체 데이터셋을 1회 학습한 상태**를 의미합니다.

### Epoch 특징
- Epoch을 여러 번 반복할수록 모델이 데이터를 더 많이 학습하게 됨
- 일반적으로 1 Epoch만으로 최적의 모델을 만들기는 어려우므로 여러 Epoch을 학습함

### Epoch 계산 공식
```python
배치 크기 (per_device_train_batch_size) = 8
Gradient Accumulation Steps (gradient_accumulation_steps) = 4
Epochs (num_train_epochs) = 5
총 데이터 개수 (len(train_data)) = 2400개
```

#### Epoch 당 Step 수 계산
```python
1 Epoch = 75 Step  # 2400 / (8 * 4) = 75
```

#### 각 Epoch 이후 저장되는 Step 번호
```
Epoch 1 → Step 75 (checkpoint-75)
Epoch 2 → Step 150 (checkpoint-150)
Epoch 3 → Step 225 (checkpoint-225)
```

---


## 4. LLM 학습에서 Step 기준과 Epoch 기준

### Step을 기준으로 하는 경우
1) 모델 파라미터 업데이트의 기본 단위
   - LLM 학습은 파라미터 업데이트(가중치 조정)를 중심으로 이루어지며, 이 과정은 Step 단위로 실행됨
   - Batch 크기만큼 데이터를 학습한 후 Step 단위로 모델이 업데이트되므로 Step이 학습의 주요 기준이 됨

2) 학습률 조정 및 스케줄링
   - 학습률(Warm-up 및 Decay) 조정은 Step 단위로 수행됨
   - Gradient Accumulation(그래디언트 누적)도 Step 단위로 이루어져, Step을 기준으로 학습 효율성을 관리해야 함

### Epoch을 기준으로 하는 경우
1) 소규모 데이터셋 기반의 파인튜닝
   - 데이터가 적은 경우, 모델이 데이터셋 전체를 몇 번 학습했는지가 중요
   - 따라서 Epoch을 기준으로 학습을 모니터링하고 성능을 평가하는 것이 적합

2) 과적합(Overfitting) 방지
   - Epoch 단위로 학습 곡선을 모니터링하면 과적합 여부를 확인할 수 있음
   - (예: Validation Loss가 특정 Epoch 이후 급격히 증가하면 과적합 가능성)

---

## 5. Sequence Length 크기와 학습

LLM이 입력으로 받을 수 있는 최대 토큰 길이를 Sequence Length라고 한다. 일반적으로 512~8,192 사이이다.

```python
max_seq_length = 2048
```

### 시퀀스 길이와 배치 크기의 관계
- 시퀀스 길이(Sequence Length)는 모델이 한 번에 처리하는 토큰(Token) 수를 의미하며, 배치 크기(Batch Size)와 함께 총 입력 토큰 수를 결정하는 중요한 요소입니다.

예를 들어, 배치 크기가 8이고 시퀀스 길이가 512인 경우:
```python
총 입력 토큰 수 = 512 * 8 = 4,096 토큰
```

- 시퀀스 길이가 길어질수록 모델이 처리해야 하는 토큰 수가 기하급수적으로 증가하며, GPU 메모리 사용량 증가(Out of Memory), 학습 속도 저하 등의 문제가 발생할 수 있음
- 따라서 시퀀스 길이가 길어질수록 GPU 메모리 부담이 커지므로, 배치 크기를 조절하는 것이 핵심

---

## 작업 유형에 따른 시퀀스 길이 설정 (Fine-tuning, RAG)

### 1. 일반적인 파인튜닝(Fine-tuning) 작업
- **권장 시퀀스 길이**: 512 ~ 4,096
- 대화형 AI(Chatbot), 텍스트 요약, 질의응답(Q/A)과 같은 언어 생성(Language Generation) 작업에 적합
- 대부분의 자연어 처리(NLP) 작업에서는 짧은 문맥 내에서 의미가 결정되기 때문
- 일반적으로 4,096 이하의 시퀀스 길이로 충분한 성능 확보 가능

### 2. RAG (Retrieval-Augmented Generation) 및 긴 문서 처리
- **권장 시퀀스 길이**: 8,192 이상
- RAG는 검색된 정보를 기반으로 응답을 생성하는 방식 → 긴 문맥 유지 필요
- 논문 요약, 법률 문서 분석, 기술 문서 이해와 같은 긴 텍스트 처리 작업에 적합
- 8,192 ~ 16,384 수준의 긴 시퀀스 지원 모델 필요


---

## 6. Optimizer (옵티마이저)

딥러닝 모델을 학습할 때, 손실 함수(Loss Function)를 정의하여 모델의 예측 값과 실제 값 사이의 차이를 측정합니다. 모델 학습의 목표는 이 손실 값을 최소화하는 것이며, 이를 최적화(Optimization)라고 합니다.

Optimizer(옵티마이저)는 이러한 최적화를 수행하는 알고리즘으로, 손실 함수의 최소값을 빠르게 찾아가는 방법을 결정하는 역할을 합니다. 즉, 옵티마이저는 모델의 가중치(Weight)를 효과적으로 업데이트하여 학습 성능을 향상시키는 핵심 요소입니다.

---

### AdamW (Adam with Weight Decay)

- Adam의 변형으로, Weight Decay(가중치 감소)를 명확하게 분리하여 과적합(Overfitting)을 방지하는 효과를 가짐
- 현재 LLM(대형 언어 모델, Large Language Model) 파인튜닝에서 가장 일반적으로 사용되는 옵티마이저

---

### 최적화 옵티마이저 비교
최근 대형 모델(LLM) 학습에서는 메모리 효율이 중요한 요소로 작용하며, 기존 AdamW를 개선한 여러 옵티마이저가 등장하였습니다. 이들은 Adam의 효율성과 성능을 유지하면서도 메모리 사용량을 줄이는 데 중점을 둡니다.

```python
optim = "adamw_torch"  # 기본 AdamW
optim = "adamw_8bit"  # 메모리 절감형 AdamW
optim = "paged_adamw_32bit"  # CPU 활용 AdamW
optim = "paged_adamw_8bit"  # 8비트 + CPU 활용 AdamW
optim = "adafactor"  # 메모리 절약형 옵티마이저
```

| 옵티마이저 | 특징 | 장점 | 단점 | 사용 추천 |
|------------|------|------|------|----------|
| **Adafactor** | 행-열 통계만 유지하여 메모리 절약 | 대형 모델 학습 가능, 자동 학습률 조정 | Adam 대비 학습 성능 낮음, 학습 불안정 가능 | 제한된 메모리 환경에서 LLM 학습 |
| **AdamW_8bit** | 8비트 정밀도 활용 | AdamW 성능 유지, 메모리 절감 | 정밀도 손실 가능, 학습 불안정 가능 | 메모리를 줄이면서 AdamW 성능 유지 |
| **Paged_AdamW** | CPU 메모리 활용 | GPU 메모리 절감 | CPU-GPU I/O 병목 발생 가능 | GPU 메모리가 부족한 대형 모델 학습 |
| **Paged_AdamW_8bit** | 8비트 + CPU 메모리 활용 | 메모리 사용 최소화 | I/O 병목 + 정밀도 손실 가능 | 극한의 메모리 절약이 필요한 경우 |

---

### 옵티마이저 선택 가이드
- **AdamW (기본 옵션)**: 가장 널리 쓰이는 옵티마이저로 성능과 안정성을 유지할 수 있음.
- **AdamW_8bit**: Adafactor만큼 메모리는 줄어들고 안정성은 높아짐. AdamW의 메모리 사용량을 줄이고 싶을 때 적합.
- **Paged_AdamW**: GPU 메모리가 부족한 환경에서 적절한 해결책.
- **Paged_AdamW_8bit**: 최대한의 메모리 절감이 필요할 때 사용 가능.
- **Adafactor**: 메모리 절약이 필수적인 경우에 유용하지만, 학습 성능 저하 가능성 존재.

---

### LoRA 학습 시 Optimizer 선택 가이드

LoRA(Low-Rank Adaptation)는 PEFT(Parameter Efficient Fine-Tuning) 기법 중 하나로, 파라미터의 일부만 학습하는 방식입니다. 이 방식은 모델의 전체 가중치를 업데이트하는 Full 파인튜닝과 비교하여 메모리 사용량이 현저히 적다는 특징이 있습니다.

따라서, LoRA 방식으로 파인튜닝을 진행할 경우, 메모리 절약형 옵티마이저(양자화 또는 페이징 적용) 사용이 필수적이지 않습니다.

#### LoRA 학습 시 권장 Optimizer
- 기본적으로는 **AdamW 사용을 권장**
  - LoRA는 메모리 사용량이 적으므로, 굳이 양자화(8bit)나 페이징 기법(Paged AdamW)을 적용할 필요 없음.
  - 기본 AdamW가 학습 안정성과 성능 면에서 가장 적절한 선택.
- **메모리가 부족한 경우 예외적으로 AdamW_8bit 또는 Paged_AdamW 사용 가능**
  - LoRA를 사용하더라도 GPU 메모리가 부족한 경우
  - → AdamW_8bit(양자화) 또는 Paged_AdamW(CPU 활용)를 고려할 수 있음.
  - 특히 LoRA를 적용하더라도 모델이 매우 크거나, 배치 크기가 커지는 경우
  - → Paged_AdamW_8bit 같은 극단적인 메모리 절약형 옵티마이저가 필요할 수 있음.


---

## 7. Learning Rate (학습률)

학습률(Learning Rate)은 Batch 크기와 함께 모델 성능에 가장 큰 영향을 미치는 하이퍼파라미터 중 하나입니다. 학습률이 너무 낮으면 학습 속도가 느려지고, 반대로 너무 높으면 발산할 위험이 있습니다.

### 학습률 조정 원칙
- 학습률이 너무 작으면 → 최소값을 찾는 데 오랜 시간이 걸리고, 국소 최적점(Local Minima)에 갇혀 최적의 성능을 내지 못할 가능성이 큼
- 학습률이 너무 크면 → 최소값을 지나쳐 손실이 수렴하지 않고 발산할 위험이 있음

### 적절한 학습률 범위
LLM을 파인튜닝할 때 적절한 학습률은 일반적으로 **1e-3 ~ 1e-6** 범위에 위치합니다.

가장 효과적인 방법은 중간값(예: `1e-4`)으로 실험을 시작한 후, 학습 결과를 바탕으로 조정하는 것입니다.

```python
learning_rate = 1e-04
```

---

## 학습률 스케줄러 (Learning Rate Scheduler)

학습이 진행되는 동안 동일한 학습률을 유지하는 것이 최선의 선택은 아닙니다. 일반적으로 학습 초기에는 빠른 수렴을 위해 비교적 높은 학습률이 필요하지만, 후반부에는 더 정밀한 최적화를 위해 학습률을 점진적으로 낮추는 것이 효과적입니다.

### 학습률 조정 흐름
- **학습 초반** → 빠른 수렴을 위해 비교적 높은 학습률 필요
- **학습 후반** → 정밀한 최적화를 위해 학습률을 점진적으로 낮추는 것이 효과적

이처럼 학습률을 동적으로 조정해주는 기법을 **Learning Rate Scheduler(학습률 스케줄러)**라고 합니다.

### 학습률 스케줄러 유형

| 유형 | 설명 |
|------|------|
| **Cosine** | 학습률을 코사인 함수 곡선을 따라 감소시키는 방식 |
| **Linear** | 일정한 비율로 학습률을 선형적으로 감소시키는 방식 |
| **Constant** | 학습률을 일정하게 유지하는 방식 (일반적인 LLM 파인튜닝에서는 권장되지 않음) |

```python
lr_scheduler_type = "linear"
lr_scheduler_type = "cosine"
lr_scheduler_type = "constant"
```

---

## 8. Warmup

Warmup은 학습률 스케줄러와 함께 사용되는 기법으로, 학습 초기에 낮은 학습률에서 시작하여 설정된 학습률까지 점진적으로 높여가며 학습하는 방법입니다.

학습을 시작할 때 너무 높은 학습률을 적용하면, 학습이 급격히 발산하거나 최적의 방향을 찾지 못할 위험이 있습니다.

이를 방지하기 위해 **초기 일정 스텝 동안 학습률을 점진적으로 증가**시키는 **Warmup 기법**을 사용합니다.

### Warmup 설정 방식
#### **1. Warmup Step**
- 학습률을 증가시키는 기간을 Step(학습 단계) 수로 지정하는 방식
- 예제: `warmup_steps=1000` → 1000 Step 동안 학습률을 점진적으로 증가

#### **2. Warmup Ratio**
- 전체 학습 과정에서 Warmup이 차지하는 비율을 지정하는 방식
- 예제: `warmup_ratio=0.1` → 전체 Step 수의 10% 동안 학습률을 점진적으로 증가

```python
warmup_ratio = 0.1  # 전체 Step의 10% 동안 Warmup 적용
warmup_steps = 100  # 1000 Step 동안 학습률 증가


---

## 9. 정밀도 (Precision)

FP32, FP16, BF16과 같은 정밀도(Precision)는 숫자를 표현하는 방식(부동소수점 형식)과 연산의 정확도를 결정하는 핵심 요소입니다.

정밀도가 높을수록 보다 넓은 범위의 숫자를 정확하게 표현할 수 있지만,
더 많은 메모리와 계산 자원이 필요하므로, 모델의 성능과 효율성을 고려하여 적절한 정밀도를 선택해야 합니다.

---

### BF16 사용을 권장

- 높은 정밀도를 요구하지 않는 LLM 파인튜닝에서는 **BF16**이 메모리와 학습 성능 면에서 가장 효율적임
- **NVIDIA Ampere(A100) 계열 이상의 GPU**에서 BF16 지원이 최적화되어 있으므로, 이를 기본적으로 사용하는 것이 가장 적절함

```python
bf16 = True  # BF16 정밀도 사용 (권장)
```

---

### FP16을 사용할 경우 손실 스케일링 적용 필요

- FP16을 사용해야 한다면 **손실 스케일링(loss scaling) 설정이 필수**
- 손실 스케일링을 적절히 조정하지 않으면 학습이 불안정해질 수 있음

```python
fp16 = True  # FP16 정밀도 사용
fp16_opt_level = "O2"  # Mixed Precision 설정
```

---

### FP32는 가급적 사용하지 않음

- **FP32는 메모리 소모가 크고 학습 속도가 느려 실질적으로 LLM 학습에는 적합하지 않음**
- 메모리 사용량이 증가하여 모델 훈련 시 비효율적일 수 있음

```python
fp32 = False  # FP32 사용 비추천
```

---

### 정밀도 선택 가이드
| 정밀도 | 장점 | 단점 | 추천 사용 |
|--------|------|------|----------|
| **BF16** | 메모리 절약, 빠른 연산, 안정적인 학습 | 일부 GPU에서 지원되지 않을 수 있음 | **대부분의 LLM 학습에 권장** |
| **FP16** | 메모리 절약, 빠른 연산 | 손실 스케일링 필요, 학습 불안정 가능 | **BF16이 지원되지 않는 환경** |
| **FP32** | 높은 정밀도 | 높은 메모리 사용량, 학습 속도 느림 | **특수한 경우 제외하고 비추천** |


## 📚 참고 문헌
- [Beomi의 KoAlpaca 프로젝트](https://github.com/Beomi/KoAlpaca?tab=readme-ov-file)
- [SKT DEVOCEAN 블로그 - PEFT 적용](https://devocean.sk.com/blog/techBoardDetail.do?ID=167242&boardType=techBlog&searchData=&searchDataMain=&page=&subIndex=&searchText=&techType=&searchDataSub=&comment=)
- [SKT DEVOCEAN 블로그 - LoRA 개념](https://devocean.sk.com/blog/techBoardDetail.do?ID=167265&boardType=techBlog)
