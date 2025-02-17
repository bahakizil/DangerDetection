# DeepSeek Danger Detection

This repository includes a sample `analysis_tool` that:
1. Reads a YOLO detection log file (`detections.txt`).
2. Invokes a **DeepSeek** model via [Ollama](https://ollama.ai/) to analyze the log.
3. Produces an **email-style** summary with the number of `Danger` detections, a short explanation, an optional warning, and a mention of a `detection_snapshot.png`.
4. Writes the final output to `analysis.txt`.

## How it Works

1. **analysis_tool** (defined in `deepseek_agent.py`):
   - Loads the YOLO detections from a file.
   - Sends a prompt to the DeepSeek model via Ollama.
   - Parses the AI response, ensuring we have bullet lines for:
     - `- Number of 'Danger' detections: X`
     - `- Brief explanation: Y`
   - Adds some fallback text if the model doesn't provide them.
   - Saves the final email-like text to `analysis.txt`.

2. **Mail Sending (Future Step)**:
   - Another agent (or Python code) can read `analysis.txt` and send it to the required email address (e.g. `kizilbaha26@gmail.com`).

## Requirements

- Python 3.8+
- `langchain-ollama` for ChatOllama
- Ollama CLI or server running locally with your chosen model (e.g. `deepseek-r1:1.5b`)
- Additional packages as listed in `requirements.txt`.

## Usage

1. Make sure you have [Ollama](https://ollama.ai) installed and running locally.
2. Pull the model:
   ```bash
   ollama pull deepseek-r1:1.5b

ollama serve

from deepseek_agent import analysis_tool

result = analysis_tool('detections.txt')
print(result)
