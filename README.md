# Stylometric-analysis-of-Japanese-AI-genereated-text [WIP]
## Step-by-step guide
### Preprocess
1. Run download_wiki40b-ja.py
2. Run preprocess_data in wiki40b mode
3. Run split_wiki40b-ja_to_parts.py
4. Run gen_AI-generated_text.py
(`src/chatgpt/client.py` is required.)
5. Run preprocess_data in chatgpt4o mode
6. Run merge_chatgpt4o_from_parts.py

### Train
1. Run train.py

### Validate
1. Run validate.py

### Predict
1. Run predict.py

## Notes
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