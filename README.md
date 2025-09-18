# ? SoM2025 Challenge Baseline

���ֿ��ṩ **SoM2025: Adapting WiFo for Wireless Physical Layer Tasks** �Ĺٷ� Baseline ʵ�֡������߿��ڴ˻������� **ģ��΢���������Ż���������** �ȸĽ���

---

## 1) ���������ݸ���

����ս���� **Ԥѵ�� WiFo-Base ģ��**�������������������

�� ����һ��ϡ�赼Ƶ�ŵ����ƣ�Channel Estimation��
- Ŀ�꣺���õ�Ƶλ�õĹ۲�������ŵ�������лָ�/�ڲ塣
- ������Դ��Quadriga
- ������UMa NLoS
- ѵ��������900
- �ŵ�ά�ȣ�4 �� 32 �� 64
- ����������ã���Ƶÿ 4 �����ز�����
- Baseline ָ�꣺NMSE = 0.318

�� �������LoS / NLoS �б�Link Type Classification��
- Ŀ�꣺�����ŵ������ж��Ƿ�Ϊ LoS��
- ������Դ��Quadriga
- ������UMi LoS + UMi NLoS
- ѵ��������300
- �ŵ�ά�ȣ�24 �� 8 �� 128
- Baseline ָ�꣺F1 = 0.828

�� ���������Ӿ��������߶�λ��Vision-aided Localization��
- Ŀ�꣺����Ӿ����ŵ�����ʵ�ֳ���λ�þ�׼��λ��
- ������Դ��SynthSoM
- ������Cross Road
- ѵ��������500
- �ŵ�ά�ȣ�1 �� 128 �� 32
- Baseline ָ�꣺MAE = 9.83

�ۺ� Baseline �ܷ֣������� 0~1����
((1 - 0.318/1.000) + 0.828 + (1 - 9.83/20)) �� (1 + 0.11)/3.6 �� 0.622

---

## 2) ���ݼ��������ʽ

���عٷ����ݼ��������Ŀ��Ŀ¼�� `./dataset/` �£����Ӽ��½ڣ���Ŀ¼�ṹ�������£�

dataset/
���� Task1/           # �ŵ��������ݣ�.npy / .pt����shape: [N, 4, 32, 64]
��  ���� X_train.mat ...
��  ���� X_val.mat   ...
���� Task2/           # LoS/NLoS �������ݣ�shape: [N, 24, 8, 128]����ǩ 0/1
��  ���� X_train.mat ...
��  ���� X_val.mat   ...
���� Task3/           # ��λ���ݣ����� shape: [N, 1, 128, 32]��Ŀ��Ϊλ������
��  ���� X_train.mat ...
��  ���� X_val.mat   ...

��ʹ���Զ����ʽ�����ڸ�����ű���������Ӧ�� `Dataset` ��ȡ�߼���

---

## 3) ���ٿ�ʼ��������������

1) ��¡�ֿ�
git clone https://github.com/SoM2025/Baseline.git
cd Baseline

2) �����뼤�������ѡһ����
# Conda
conda create -n som2025 python=3.9 -y
conda activate som2025
# �� venv
python -m venv .venv && source .venv/bin/activate    # Windows ʹ�� .venv\Scripts\activate

3) ��װ����
pip install -r requirements.txt
 
---

## 4) ���عٷ����ݼ�

������µ�ַ���ز���ѹ����Ŀ��Ŀ¼�� ./dataset/ �£�
���ٷ����ݼ����ء�https://YOUR-DATA-HOST/som2025-dataset
����Ϊ˽�����أ����滻Ϊʵ�����ӣ����� Task1/Task2/Task3 ��Ŀ¼�ṹ���䣩

---

## 5) ѵ��������

�ֱ��������нű�ѵ���������񣨻��Զ��� ./logs �±���ģ����������

# ����һ���ŵ�����
python main_fine_tune_T1.py

# �������LoS/NLoS �б�
python main_fine_tune_T2.py

# ���������Ӿ�������λ
python main_fine_tune_T3.py
 
---

## 6) �ύ�淶��submission��

�� �ύ�ļ�һ�������� JSON���� submission_demo.json ��ʽһ�£�
�ļ�����submission.json
��ʽ��
{
  "Task1": Array_t1,  // �ŵ����ƣ�Ԥ����󣬽��� shape [N, 4, 32, 64]
  "Task2": Array_t2,  // LoS/NLoS��Ԥ���ǩ���飨0/1����shape [N] �� [N, 1]
  "Task3": Array_t3   // ��λ��λ��Ԥ�����Ĭ�� shape [N, 2] ��ʾ (x, y)
}

ע�⣺
- JSON �е�����ӦΪ�����л���Ƕ���б�list of lists����
- ��Ҫ���� NaN/Inf����ʹ�� float/������

�� �ύ�ļ�����ƽ��ѵ�������������
- ͳ�Ʒ�����ѵ��ʱ�ɼ�¼ "ģ���� requires_grad �Ĳ�������" ��Ϊ��ѵ������������ʹ�� LoRA/Adapter �ȣ���ͳ�ƿ�ѵ�����֣�����ͬ�׶���������ģ�б仯���ɰ�ѵ���ڼ��ʱ��/epoch ��Ȩƽ����
 

- �ο�ͳ�ƽű���PyTorch����
# tools/count_params.py
import torch
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
 
---

## 7) ���֤����л

- Ԥѵ��������WiFo����Դ��ַ��ο���Ŀ��ҳ/�������ã�
- �� Baseline ������ SoM2025 ��ս��ѧ������Ŀ�ġ�
- �������Ļ򱨸���ʹ�ñ���Ŀ��������������л SoM2025 ��ί���� WiFo ��Ŀ��

ף��λ�� SoM2025 ��ս����ȡ������ɼ���
