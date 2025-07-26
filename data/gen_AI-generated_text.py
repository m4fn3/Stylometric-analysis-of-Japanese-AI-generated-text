import pyprojroot
import json
from tqdm import tqdm
import sys
import os
import asyncio

base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))
sys.path.append(str(base_path))

from src.chatgpt.client import AsyncChatGPTClient
from src.preprocessor import preprocess_text

client = AsyncChatGPTClient()
semaphore = asyncio.Semaphore(10) # 最大同時に10リクエストまで
length = 400

async def fetch_response(title: str, first_sentence: str) -> tuple[str, str]:
    prompt = f"以下の条件に基づいて文章を生成してください。\n1. 「{title}」をテーマとした文章であること\n2. 生成する文章は日本語で記述すること\n3. 文章の長さは400〜500文字程度にすること\n4. 次の括弧内に示す文章に続けて文章を記述すること: 「{first_sentence}」\n5. 回答には4.で示した文章は含めず、新たに生成した後続文のみを提供し、それ以外の内容は一切回答に含めないこと"
    async with semaphore:
        try:
            res = await client.get_response(prompt)
            print("[*] Response received for title:", title)
            return title, res
        except Exception as e: # 1度までリトライする
            print(f"[!] Error fetching response for title '{title}': {e}")
            try:
                await asyncio.sleep(5)
                res = await client.get_response(prompt)
                print("[*] Response received for title:", title)
                return title, res
            except Exception as e:
                print(f"[!] Error fetching response for title '{title}': {e}")
                return title, ""

async def main():
    for split in tqdm(["test_part1", "test_part2", "validation_part1", "validation_part2"]):
        data = {} # {title: text}
        with open(base_path / f"data/preprocessed/wiki40b-ja/parts/wiki40b-ja_{split}.json", "r", encoding="utf-8") as f:
            org = json.load(f)
            
            tasks = []
            for title, text in org.items():
                first_sentence = text.split("。")[0]+ "。"
                tasks.append(fetch_response(title, first_sentence))
            
            for i, task in enumerate(asyncio.as_completed(tasks)):
                title, resp = await task
                data[title] = preprocess_text(resp, length)
                if i % 100 == 0:
                    print(f"[o] Processed {i} responses and saving partial data...")
                    with open(base_path / f"data/preprocessed/chatgpt4o/parts/chatgpt4o_{split}_partial.json", "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
        with open(base_path / f"data/preprocessed/chatgpt4o/parts/chatgpt4o_{split}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.remove(base_path / f"data/preprocessed/chatgpt4o/parts/chatgpt4o_{split}_partial.json")
        print(f"[o] Completed chatgpt4o_{split}!")
            
if __name__ == "__main__":
    asyncio.run(main())
                
            
                
                    
                               