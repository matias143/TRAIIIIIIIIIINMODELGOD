from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json

# Cargar el modelo y el tokenizer entrenados
model_name = "./trained_model"  # Cambia esto si el modelo está en otra ubicación
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Cargar un modelo para calcular la similitud semántica
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Cargar el archivo JSON con los contextos
with open("dataset_contexts.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)["contexts"]

# Crear embeddings para los contextos
context_embeddings = embedding_model.encode([entry["context"] for entry in dataset])

def find_relevant_context(question):
    """Encuentra el contexto más relevante para la pregunta usando similitud semántica."""
    question_embedding = embedding_model.encode([question])
    similarities = cosine_similarity(question_embedding, context_embeddings)
    best_index = similarities.argmax()

    # Filtrar solo si la similitud supera un umbral
    threshold = 0.6  # Ajusta este valor según tu necesidad
    if similarities[0, best_index] < threshold:
        return None  # No hay contexto relevante

    return dataset[best_index]["context"]


def answer_question(question):
    """Genera una respuesta basada en la pregunta y el contexto más relevante."""
    context = find_relevant_context(question)
    if not context:
        return "No se encontró un contexto relevante para la pregunta."
    
    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation="only_second",
        stride=128,
        padding="max_length",
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start_index = torch.argmax(outputs.start_logits)
    answer_end_index = torch.argmax(outputs.end_logits) + 1

    # Validar índices
    if answer_start_index >= answer_end_index or answer_end_index > len(inputs.input_ids[0]):
        return "No se encontró una respuesta relevante."
    
    # Decodificar la respuesta
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start_index:answer_end_index])
    )
    return answer.strip() if answer.strip() else "No se encontró una respuesta relevante."

def chat():
    """Interfaz de chat en terminal."""
    print("=== Chat con el modelo ===")
    print("Escribe 'salir' para terminar.\n")
    
    while True:
        question = input("Pregunta: ")
        if question.lower() == "salir":
            break
        
        # Obtener respuesta
        answer = answer_question(question)
        print(f"Respuesta: {answer}\n")

if __name__ == "__main__":
    chat()
