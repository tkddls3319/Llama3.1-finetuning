# 02_Scripts폴더의 SFT_meta__Llama-3.1-8B-Instruct.ipynb을 실행 시켜 사용해보세요

# 🌟 Llama 3.1 LoRA 기반 PEFT 적용 가이드

LoRA (Low-Rank Adaptation)는 LLM(대형 언어 모델)의 일부 파라미터만 미세 조정하여 메모리와 계산량을 절약하는 파인튜닝 기법입니다.

Llama 3.1 모델에 LoRA를 적용하여 PEFT(Parameter-Efficient Fine-Tuning)를 수행하는 방법을 이론과 실전 코드 예제를 통해 단계별로 설명합니다.

---

## 📌 1. PEFT(LoRA)란 무엇인가?

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

## 📌 2. Batch, Step, Epoch 개념

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

### 🏷️ 1) Batch (배치)
- 한 번의 학습 단계에서 사용되는 데이터 샘플의 개수입니다.
- **Batch Size**는 학습 속도, 메모리 사용량, 일반화 성능 등에 영향을 줍니다.

### 🏷️ 2) Step (학습 스텝)
- 모델이 한 번 업데이트되는 과정을 의미합니다.
- 예: 데이터셋 크기가 10,000개이고, Batch 크기가 10이면 `10,000 / 10 = 1,000 Step`이 진행됩니다.

### 🏷️ 3) Epoch (에포크)
- 학습 전체 데이터셋을 1회 학습한 상태를 1 Epoch이라고 합니다.
- Epoch 수를 조절하여 학습 성능을 최적화할 수 있습니다.

---

## 📌 3. Sequence Length 크기와 학습

LLM이 입력으로 받을 수 있는 최대 토큰 길이를 **Sequence Length**라고 합니다.
```python
max_seq_length = 2048
```

- **시퀀스 길이가 길어질수록 GPU 메모리 사용량 증가**
- **배치 크기 조절이 중요** (예: 배치 크기가 8, 시퀀스 길이가 512 → 총 입력 토큰 수 = 512 * 8 = 4,096 토큰)

### 🔹 작업 유형에 따른 시퀀스 길이 설정
| 작업 유형 | 권장 시퀀스 길이 |
|-----------|--------------|
| 일반적인 파인튜닝 (대화형 AI, 텍스트 요약) | 512 ~ 4,096 |
| RAG (긴 문서 처리) | 8,192 이상 |

---

## 📌 4. Optimizer (옵티마이저)

Optimizer(옵티마이저)는 손실 함수를 최소화하는 역할을 합니다.

### 🔹 AdamW (Adam with Weight Decay)
- LLM 학습에서 가장 많이 사용되는 옵티마이저

---

## 📌 5. Learning Rate (학습률) 설정
```python
lr_scheduler_type="linear"  # 선형 감소
lr_scheduler_type="cosine"  # 코사인 감소
lr_scheduler_type="constant"  # 일정 유지 (권장되지 않음)
```

---

## 📌 6. Warmup 설정
```python
warmup_ratio=0.1  # 전체 Step의 10% 동안 Warmup 적용
warmup_steps=100  # 100 Step 동안 학습률 증가
```

---

## 📌 7. Precision (정밀도) 설정
```python
bf16=True  # BF16 정밀도 사용 (권장)
```

---

## 📚 참고 문헌
- [Beomi의 KoAlpaca 프로젝트](https://github.com/Beomi/KoAlpaca?tab=readme-ov-file)
- [SKT DEVOCEAN 블로그 - PEFT 적용](https://devocean.sk.com/blog/techBoardDetail.do?ID=167242&boardType=techBlog&searchData=&searchDataMain=&page=&subIndex=&searchText=&techType=&searchDataSub=&comment=)
- [SKT DEVOCEAN 블로그 - LoRA 개념](https://devocean.sk.com/blog/techBoardDetail.do?ID=167265&boardType=techBlog)
