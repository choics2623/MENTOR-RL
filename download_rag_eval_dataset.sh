DOWNLOAD_DIR="./eval_dataset/original"

mkdir -p ${DOWNLOAD_DIR}/hotpotqa
wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/hotpotqa/train.jsonl -O ${DOWNLOAD_DIR}/hotpotqa/train.jsonl
wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/hotpotqa/dev.jsonl -O ${DOWNLOAD_DIR}/hotpotqa/dev.jsonl

mkdir -p ${DOWNLOAD_DIR}/2wikimultihopqa
wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/2wikimultihopqa/train.jsonl -O ${DOWNLOAD_DIR}/2wikimultihopqa/train.jsonl
wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/2wikimultihopqa/dev.jsonl -O ${DOWNLOAD_DIR}/2wikimultihopqa/dev.jsonl

mkdir -p ${DOWNLOAD_DIR}/musique
wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/musique/train.jsonl -O ${DOWNLOAD_DIR}/musique/train.jsonl
wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/musique/dev.jsonl -O ${DOWNLOAD_DIR}/musique/dev.jsonl

mkdir -p ${DOWNLOAD_DIR}/bamboogle
wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/bamboogle/test.jsonl -O ${DOWNLOAD_DIR}/bamboogle/test.jsonl