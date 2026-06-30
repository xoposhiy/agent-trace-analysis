from datasets import load_dataset
logs = load_dataset("SALT-NLP/SWE-chat", "session_logs", split="train")
print(logs[0])