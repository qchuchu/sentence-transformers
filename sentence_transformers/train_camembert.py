from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.models import CamemBERT, Pooling
from sentence_transformers.readers import NLIDataReader
from sentence_transformers.losses import SoftmaxLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import math
import logging
from datetime import datetime

# Use CamemBERT for mapping tokens to embeddings
model_name = 'camembert-base'
word_embedding_model = CamemBERT(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension(),
                        pooling_mode_mean_tokens=True,
                        pooling_mode_cls_token=False,
                        pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = 'output/training_fquad_'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

fquad_reader = NLIDataReader('datasets/FQuad')
batch_size = 4
train_num_labels = fquad_reader.get_num_labels()

train_data = SentencesDataset(fquad_reader.get_examples('train.gz'), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = SoftmaxLoss(model=model,
                         sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                         num_labels=train_num_labels)

logging.info("Read FQuad dev dataset")
dev_data = SentencesDataset(examples=fquad_reader.get_examples('dev.gz'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

# Configure the training
num_epochs = 1
warmup_steps = math.ceil(len(train_data) * num_epochs / batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

if __name__ == '__main__':
    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path
              )