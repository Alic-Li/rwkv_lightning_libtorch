# rwkv_lightning_LibTorch 🕊️ ⚡
RWKV Batch infer backend Base on [Albatross](https://github.com/BlinkDL/Albatross) 🕊️ and [rwkv_lightning](https://github.com/RWKV-Vibe/rwkv_lightning) and [drogon](https://github.com/drogonframework/drogon) 🐉
- Thanks to [Rapid-Sampling](https://github.com/Triang-jyed-driung/Rapid-Sampling) Kernel From [Triang-jyed-driung](https://github.com/Triang-jyed-driung), it also have native HIP kerel compatible with ROCm😎

## Export Weights

The original Python `.pth` checkpoint is a `torch.save` state dict. It must be
preprocessed once and exported to a single `.pt` file first.

The exported `.pt` is not a regular `torch.save` file. It is a pickle container
designed specifically for C++ `torch::pickle_load`.

```bash
cd rwkv_lightning_libtorch

python export_rwkv7_weights.py \
  --model /mnt/pc411_data/rwkv_translate/rwkv7-g1d-2.9b-20260131-ctx8192 \
  --out weights/rwkv7-g1d-2.9b.pt
```

This script applies the same weight preprocessing used by Albatross:

- transpose selected matrices
- apply `squeeze`
- flatten `att.r_k`
- apply `blocks.0.ln0` to `emb.weight`
- rename `blocks.0.att.v0/v1/v2` to `a0/a1/a2`

The exported `.pt` is a single-file tensor list:

- the first tensor stores model metadata
- all remaining tensors are stored in a fixed order

## Build
**Install the Dorgon framework**

- 🐉 https://github.com/drogonframework/drogon 

**For Nvidia CUDA**
```bash
cmake -S . -B build -DRWKV_BACKEND=cuda -DTorch_DIR=/home/alic-li/python_env/py312/lib/python3.12/site-packages/torch/share/cmake/Torch
cmake --build ./build -j 8 --target rwkv_backend_support benchmark rwkv_lightning
```
**For AMD ROCm**

```bash
cmake -S . -B build -DRWKV_BACKEND=hip -DTorch_DIR=/home/alic-li/python_env/py312/lib/python3.12/site-packages/torch/share/cmake/Torch
cmake --build ./build -j 8 --target rwkv_backend_support benchmark rwkv_lightning
```

## Usage
```bash
./build/rwkv_lightning --port <your model path> --port <your port number> --password <your password>
```
- if no password, you can do not add ```--password``` flag, "--port" default port is 8000.

## Run simple benchmark

Single-sample decode:

```bash
./build/benchmark \
    --weights /mnt/sda1/rwkv_weights/libtorch_pt/rwkv7-g1d-2.9b.pt \
    --decode-prompt "User: simulate SpaceX mars landing using python\n\nAssistant: <think" \
    --decode-steps 512 \
    --decode-temp 1.0
```

Batch decode. `--batch-prompts` uses `;` to separate multiple prompt token
sequences:

```bash
./build/benchmark \
    --weights /mnt/sda1/rwkv_weights/libtorch_pt/rwkv7-g1d-2.9b.pt \
    --batch-prompts-text "User: simulate SpaceX mars landing using python\n\nAssistant: <think;User: simulate SpaceX mars landing using python\n\nAssistant: <think;User: simulate SpaceX mars landing using python\n\nAssistant: <think;User: simulate SpaceX mars landing using python\n\nAssistant: <think;User: simulate SpaceX mars landing using python\n\nAssistant: <think;User: simulate SpaceX mars landing using python\n\nAssistant: <think;User: simulate SpaceX mars landing using python\n\nAssistant: <think;User: simulate SpaceX mars landing using python\n\nAssistant: <think" \
    --batch-size 8 \
    --batch-steps 32 \
    --batch-temp 1.2
```

If only one `--batch-prompts` sequence is provided while `--batch-size > 1`,
the program will automatically replicate that prompt across the batch.

## API Docs 

### FIM ( For RWKV7_G1c series model )

<details>
<summary>curl examples</summary>

**Batch stream inference using [FIM/v1/batch-FIM interface]**

```bash
curl -X POST http://localhost:8000/FIM/v1/batch-FIM \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": [
      "The rain had stopped, but the street still glistened like a river of broken glass.",
      "She wasn’t sure why she’d come back.",
      "A cat darted from the alley,"
    ],
    "suffix": [
      "though everyone knew Mr. Ellis hadn’t opened that door in three years.",
      "sounding almost like her name.",
      "And then, from inside, a single lamp clicked on."
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'
```

**Batch inference using [FIM/v1/batch-FIM interface]**

```bash
curl -X POST http://localhost:8000/FIM/v1/batch-FIM \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": [
      "The rain had stopped, but the street still glistened like a river of broken glass.",
      "She wasn’t sure why she’d come back.",
      "A cat darted from the alley,"
    ],
    "suffix": [
      "though everyone knew Mr. Ellis hadn’t opened that door in three years.",
      "sounding almost like her name.",
      "And then, from inside, a single lamp clicked on."
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'
```

**BachSize == 1 Super Fast stream Infer with CUDA graph**

make sure 
    "temperature": 0.8,
    "top_k": 1,
    "top_p": 0,
    "alpha_presence": 0,
    "alpha_frequency": 0,
    "alpha_decay": 0.96,
For Text Generation

make sure 
    "temperature": 0.5,
    "top_k": 100,
    "top_p": 0.5,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
For Coding

```bash
curl -X POST http://localhost:8000/FIM/v1/batch-FIM \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": [
      "A cat darted from the alley,"
    ],
    "suffix": [
      "And then, from inside, a single lamp clicked on."
    ],
    "stop_tokens": [0, 261, 24281],
    "max_tokens": 4096,
    "chunk_size": 64,
    "temperature": 0.8,
    "top_k": 1,
    "top_p": 0,
    "alpha_presence": 0,
    "alpha_frequency": 0,
    "alpha_decay": 0.996,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'
```
**BachSize == 1 Super Fast Infer with CUDA graph**
```bash
curl -X POST http://localhost:8000/FIM/v1/batch-FIM \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": [
      "A cat darted from the alley,"
    ],
    "suffix": [
      "And then, from inside, a single lamp clicked on."
    ],
    "max_tokens": 4096,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "top_k": 1,
    "top_p": 0,
    "alpha_presence": 0,
    "alpha_frequency": 0,
    "alpha_decay": 0.996,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'
```

</details>


### 1. Batch synchronous Translate 

<details>
<summary>curl examples</summary>

**Compatible with immersive translation custom API**
**--- Very stable 🚀 ---** 
```bash
curl -X POST http://localhost:8000/translate/v1/batch-translate \
         -H "Content-Type: application/json" \
         -d '{
           "source_lang": "en",
           "target_lang": "zh-CN",
           "text_list": ["Hello world!", "Good morning"]
         }'
```
```bash
curl -X POST http://localhost:8000/translate/v1/batch-translate \
         -H "Content-Type: application/json" \
         -d '{
           "source_lang": "zh-CN",
           "target_lang": "en",
           "text_list": ["你好世界", "早上好"]
         }'
```
</details>


### 2. ```v1/chat/completions```  [Support all decode parameters]

<details>
<summary>curl examples</summary>

**--- Very stable 🚀 ---** 
- Streaming synchronous batch processing 
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      "English: After a blissful two weeks, Jane encounters Rochester in the gardens. He invites her to walk with him, and Jane, caught off guard, accepts. Rochester confides that he has finally decided to marry Blanche Ingram and tells Jane that he knows of an available governess position in Ireland that she could take.\n\nChinese:",
      "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:"
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'
```
- Non-streaming synchronous batch processing
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      "English: After a blissful two weeks, Jane encounters Rochester in the gardens. He invites her to walk with him, and Jane, caught off guard, accepts. Rochester confides that he has finally decided to marry Blanche Ingram and tells Jane that he knows of an available governess position in Ireland that she could take.\n\nChinese:",
      "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:"
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'
```

</details>


### 3. ```v2/chat/completions``` [Support all decode parameters]

<details>
<summary>curl examples</summary>

**--- Very stable 🚀 ---** 
- Streaming synchronous continuous batching processing 
```bash
curl -X POST http://localhost:8000/v2/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "contents": [
      "English: After a blissful two weeks, Jane encounters Rochester in the gardens. He invites her to walk with him, and Jane, caught off guard, accepts. Rochester confides that he has finally decided to marry Blanche Ingram and tells Jane that he knows of an available governess position in Ireland that she could take.\n\nChinese:",
      "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:"
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 1.0,
    "top_k": 1,
    "top_p": 0.3,
    "pad_zero": true,
    "alpha_presence": 0.8,
    "alpha_frequency": 0.8,
    "alpha_decay": 0.996,
    "chunk_size": 128,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'
```
- Non-streaming synchronous continuous batching processing
```bash
curl -X POST http://localhost:8000/v2/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      "English: After a blissful two weeks, Jane encounters Rochester in the gardens. He invites her to walk with him, and Jane, caught off guard, accepts. Rochester confides that he has finally decided to marry Blanche Ingram and tells Jane that he knows of an available governess position in Ireland that she could take.\n\nChinese:",
      "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:"
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 1.0,
    "top_k": 1,
    "top_p": 0.3,
    "pad_zero": true,
    "alpha_presence": 0.8,
    "alpha_frequency": 0.8,
    "alpha_decay": 0.996,
    "chunk_size": 32,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'
```

</details>


### 4. ```v3/chat/completions``` [Support all decode parameters]

<details>
<summary>curl examples</summary>

- Streaming asynchronous batch processing With CUDA Graph For Bsz=1
```bash
curl -X POST http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 1.0,
    "top_k": 1,
    "top_p": 0.3,
    "pad_zero": true,
    "alpha_presence": 0.8,
    "alpha_frequency": 0.8,
    "alpha_decay": 0.996,
    "chunk_size": 128,
    "stream": true,
    "enable_think": true,
    "password": "rwkv7_7.2b"
  }'
```
- Non-streaming asynchronous batch processing With CUDA Graph For Bsz=1
```bash
curl -X POST http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 1.0,
    "top_k": 1,
    "top_p": 0.3,
    "pad_zero": true,
    "alpha_presence": 0.8,
    "alpha_frequency": 0.8,
    "alpha_decay": 0.996,
    "chunk_size": 128,
    "stream": false,
    "enable_think": true,
    "password": "rwkv7_7.2b"
  }'
```

</details>


### 5. ```state/chat/completions``` [Support state cache manager] 😜

#### Have 3 Levels Cache design 🤓
- **L1 cache(VRAM) 16**
- **L2 cache(RAM) 32**
- **L3 cache(Sqlite3 database)**
#### The all cached state will be stored in the database when shout down the server 😋
- could modify the cache size in ```./state_pool.py``` in line 14-16

***Need to add a unique "session_id": "XXX" in the request body as a unique identifier for each session***👆

**ONLY support for bsz = 1 one session** 🤫

<details>
<summary>curl examples</summary>

- Streaming asynchronous batch processing With CUDA Graph For Bsz=1
```bash
curl -X POST http://localhost:8000/state/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "contents": [
      "User: What should we eat for dinner? Any brief suggestions?\\n\\nAssistant: <think>\\n</think>\\n"
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": true,
    "chunk_size": 128,
    "password": "rwkv7_7.2b",
    "session_id": "session_one"
  }'
```
- Non-streaming asynchronous batch processing With CUDA Graph For Bsz=1
```bash
curl -X POST http://localhost:8000/state/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
    "contents": [
      "User: What should we eat for dinner? Any brief suggestions?\\n\\nAssistant: <think>\\n</think>\\n"
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": false,
    "password": "rwkv7_7.2b",
    "session_id": "session_one"
  }'
```

</details>


### 6. **State Management API** [Support state cache manager] 😜 

#### Use ```state/status```  Interface to check the state pool status of a session

<details>
<summary>curl examples</summary>

```bash
curl -X POST http://localhost:8000/state/status \
  -H "Content-Type: application/json" \
  -d '{
    "password": "rwkv7_7.2b"
  }'
```

</details>

#### Use ```state/delete```  Interface to delete the state of a session

<details>
<summary>curl examples</summary>


```bash
curl -X POST http://localhost:8000/state/delete \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your_session_id_to_delete",
    "password": "rwkv7_7.2b"
  }'
```

</details>

### 7. ```/openai/v1/chat/completions``` [Open AI format support]
- "could be used for chat fronted which OpenAI API compatibility. Such as Cherry studio."
<details>
<summary>curl examples</summary>

- Streaming asynchronous Open AI API
```bash
curl -X POST 'http://localhost:8000/openai/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer your-password-if-set' \
  --data '{
    "model": "rwkv7",
    "messages": [
      {"role": "user", "content": "please tell me about the history of artificial intelligence"}
    ],
    "top_p": 0.6,
    "max_tokens": 2048,
    "temperature": 0.8,
    "stream": true
  }'
```
- Non-streaming asynchronous Open AI API
```bash
curl -X POST 'http://localhost:8000/openai/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer your-password-if-set' \
  --data '{
    "model": "rwkv7",
    "messages": [
      {"role": "user", "content": "please tell me about the history of artificial intelligence"}
    ],
    "top_p": 0.6,
    "max_tokens": 2048,
    "temperature": 0.8,
    "stream": false
  }'
```
</details>

### 8. ```/big_batch/completions```  [Only Support noise & temperature decode parameters]

<details>
<summary>curl examples</summary>

**The Fastest Batch Processing API 🚀** 
- Streaming synchronous batch processing 
```bash
curl -X POST 'http://localhost:8000/big_batch/completions' \
  --header 'Content-Type: application/json' \
  --data '{
    "contents": [
      "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:",
      "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:"
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 1.0,
    "chunk_size": 8,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'
```
</details>