import transformers
import torch 

DEVICE = "cuda" if torch.cuda.is_available() else "gpu"
MAX_LEN = 512
TRAIN_BATCH_SIZE = 3
VALID_BATCH_SIZE = 1
EPOCHS = 10
ACCUMULATION = 2
BERT_PATH = "C:/Users/12158/Desktop/BERTQ&A/input/bert_base_uncased/"
MODEL_PATH = "model.bin"
TRAINING_FILE = "C:/Users/12158/Desktop/BERTQ&A/input/IMDB Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
                BERT_PATH,
                do_lower_case=True)
