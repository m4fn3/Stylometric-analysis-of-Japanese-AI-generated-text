# Stylometric-analysis-of-Japanese-AI-generated-text
This project aims to reveal stylometric features of Japanese AI-generated text, focusing on distinguishing between human-written and AI-generated content using Random Forest model.

## Step-by-step guide
### Preprocess
1. Run `download_wiki40b-ja.py`
2. Run `preprocess_data.py` in wiki40b mode
3. Run `split_wiki40b-ja_to_parts.py`
4. Run `gen_AI-generated_text.py`
(`src/chatgpt/client.py` is required.)
5. Run `preprocess_data.py` in chatgpt4o mode
6. Run `merge_chatgpt4o_from_parts.py`
7. Run `build_features.py`

### Train
1. Run `train.py`

### Validate
1. Run `validate.py`

### Predict
1. Run `predict.py`

## Notes
- Label 0 is for human-written text, and label 1 is for AI-generated text.
- Merged raw / preprocessed data is excluded due to its size. Please refer to the splitted data in `data/preprocessed/chatgpt4o/parts/`.
- `src/chatgpt/client.py` is excluded. The structure is like below:
```python
class AsyncChatGPTClient:
    def __init__(self):
        pass
        
    async def get_response(self, text: str) -> str:
        pass
        return "response"
```

## Directory Structure
```yaml
├── README.md
├── data
│   ├── download_wiki40b-ja.py # Download wiki40b-ja dataset
│   ├── features
│   │   ├── extract_certain_features.py # Extract certain feature vectors for ablation study
│   │   ├── {split}.pkl # Feature vectors
│   ├── gen_AI-generated_text.py # Generate AI-generated text based on preprocessed wiki40b-ja data
│   ├── preprocessed
│   │   ├── chatgpt4o
│   │   │   ├── chatgpt4o_{split}.json
│   │   │   └── parts # Splitted data of AI-generated texts
│   │   └── wiki40b-ja
│   │       ├── parts # Splitted data of wiki40b-ja
│   │       ├── split_wiki40b-ja_to_parts.py
│   │       ├── wiki40b-ja_{split}.json
│   └── raw
│       ├── chatgpt4o
│       │   └── parts 
│       ├── wiki40b-ja_{split}.json
├── models
│   ├── RF_dNone_n100.pkl
├── src
│   ├── chatgpt
│   │   └── client.py # ChatGPT API client
│   ├── feature_extractor.py # Core implementation of vectorizing features
│   ├── misc
│   │   ├── feature_ablation.py # Create a list of feature indices to exclude for ablation study
│   │   ├── feature_importance.py # Visualize feature importance with matplotlib
│   │   ├── feature_names.json # Complete list of feature names
│   │   ├── feature_names.py # Create a list of feature names 
│   │   └── feature_statistics.py # Calculate statistics of a certain feature
│   ├── predictor.py # Core implementation of prediction
│   ├── preprocessor.py # Utility for preprocessing
│   ├── trainer.py # Core implementation of training
│   └── validator.py # Core implementation of validation
├── build_features.py # Build feature vectors from preprocessed data
├── predict.py # Interface for prediction
├── preprocess_data.py # Preprocess raw data
├── requirements.txt
├── train.py # Interface for training
└── validate.py # Interface for validation
```

## Acknowledgements
- Wiki-40B : https://research.google/pubs/wiki-40b-multilingual-language-model-dataset/

- wiki40b-ja : https://huggingface.co/datasets/range3/wiki40b-ja