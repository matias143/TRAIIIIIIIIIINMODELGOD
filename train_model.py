from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import json

# Cargar el dataset desde un archivo JSON en formato SQuAD
def load_squad_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        squad_data = json.load(f)
    data = []
    for entry in squad_data["data"]:
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                if not qa["is_impossible"]:
                    answer = qa["answers"][0]
                    data.append({
                        "context": context,
                        "question": question,
                        "answer_text": answer["text"],
                        "answer_start": answer["answer_start"]
                    })
    return Dataset.from_list(data)

dataset_path = "dataset_squad_format.json"  # Cambia esto a la ruta de tu archivo
dataset = load_squad_dataset(dataset_path)

# Dividir el dataset en entrenamiento y validación
split = dataset.train_test_split(test_size=0.2)
train_dataset = split["train"]
eval_dataset = split["test"]

# Preparar el tokenizer y el modelo
model_name = "timpal0l/mdeberta-v3-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Preprocesar el dataset
def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    start_positions = []
    end_positions = []
    for i, offset in enumerate(inputs["offset_mapping"]):
        start_char = examples["answer_start"][i]
        end_char = start_char + len(examples["answer_text"][i])
        sequence_ids = inputs.sequence_ids(i)

        # Mapear las posiciones de inicio y fin al token correspondiente
        token_start = 0
        while sequence_ids[token_start] != 1:
            token_start += 1
        token_end = len(offset) - 1
        while sequence_ids[token_end] != 1:
            token_end -= 1

        if offset[token_start][0] <= start_char and offset[token_end][1] >= end_char:
            while token_start < len(offset) and offset[token_start][0] <= start_char:
                token_start += 1
            start_positions.append(token_start - 1)
            while offset[token_end][1] >= end_char:
                token_end -= 1
            end_positions.append(token_end + 1)
        else:
            start_positions.append(0)
            end_positions.append(0)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Aplicar preprocesamiento
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

# Configuración de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=10,
    logging_dir="./logs",
)

# Entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo
trainer.save_model("./trained_model")
