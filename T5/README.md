# CoNaLa using T5

## 1. Setup
- make virtual environment and install required packages
    ```bash
    pip install -r requirements.txt
    ```

## 2. Fetch data
- Get CoNaLa dataset from the repository.
    ```bash
    bash fetch _data.sh
    ```

## 3. Train model
- Move to the ```src``` directory and run fine-tuning.
- You can modify the ```config.yaml``` file for different hyperparameter setting.
  ```bash
  mv src
  python finetune_t5.py --backbone_model t5-large --repeat 0 --n_gpu 1
  ```
  
## 4. Inference test dataset
- Generate results for the test sets.
    ```bash
    python inference_t5.py --backbone_model t5-large --repeat 0
    ```
  
## 5. Evaluate performance
- Calculate automatic evaluation metrics.
  ```bash
  python evaluate_result.py --backbone_model t5-large --repeat 0
  ```
