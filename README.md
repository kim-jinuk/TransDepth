## TransDepth

### 0. 개요
  본 연구는 투명도, 빛의 굴절 흐름(refractive flow) 및 반사(reflect) 혹은 감쇠(attenuation) 등의 투명 객체 특성에 대해 분석하고 이를 기반으로 지식 증류(Knowledge Distillation) 기법을 응용하여 배경의 간섭으로부터 강인한 투명 객체 3차원 정보 추정 방법을 제안한다.

  ![teaser](Asset/Qualitative.png)
  ![teaser](Asset/quantitative.png)
  
### 1. 데이터셋 파일 경로

  코드 경로: ./ <br>
  데이터 경로: ./datasets <br>
  학습된 모델의 경로: ./weights <br>
  샘플 경로: ./samples <br>

****


### 2. 설치

      conda create --name transdepth python=3.10 -y; conda activate transdepth
      conda install pip; pip install --upgrade pip
      pip install keras
      pip install pillow
      pip install matplotlib
      pip install scikit-learn
      pip install scikit-image
      pip install opencv-python
      pip install pydot
      
****

### 3. 학습

      python train_FKD.py --data nyu --gpus 4 --bs 8

****


### 4. 참고

* DenseDepth: [https://github.com/NVlabs/instant-ngp.git](https://github.com/ialhashim/DenseDepth.git)
      
      
<br>
