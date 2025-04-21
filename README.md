## TransDepth

### 0. 개요
  본 프로젝트는 투명 객체에 대해 정확한 깊이 추정을 위해 지식 증류 기법을 응용하여 구현하였습니다.
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
