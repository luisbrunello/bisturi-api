from flask import Flask, request, jsonify
import numpy as np
import faiss
import json
from openai import OpenAI
import os

app = Flask(__name__)

# 🔐 Substitua pela sua chave da OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# 📥 Carrega os dados do livro
with open("referencias.json", "r", encoding="utf-8") as f:
    referencias = json.load(f)

index = faiss.read_index("indice_capitulos.faiss")

# 🔍 Função para gerar embedding
def gerar_embedding(texto):
    response = client.embeddings.create(
        input=texto,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding)

# 🚀 Rota principal da API
@app.route("/perguntar", methods=["POST"])
def perguntar():
    dados = request.get_json()
    pergunta = dados.get("pergunta")

    if not pergunta:
        return jsonify({"erro": "Pergunta não fornecida."}), 400

    # Busca FAISS
    embedding = gerar_embedding(pergunta).reshape(1, -1)
    _, indices = index.search(embedding, 3)

    contexto = ""
    capitulos_usados = set()

    for i in indices[0]:
        trecho = referencias[i]["texto"]
        capitulo = referencias[i]["capitulo"]
        contexto += f"\n[{capitulo}]\n{trecho}\n"
        capitulos_usados.add(capitulo)

    # Prompt para o GPT
    prompt = f"""
Responda com base apenas no contexto abaixo.
Se não encontrar a resposta, diga "Essa informação não está disponível no material fornecido."

Contexto:
{contexto}

Pergunta: {pergunta}
"""

    resposta = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": "Você é um assistente médico especializado em cirurgia geral."},
            {"role": "user", "content": prompt}
        ]
    )

    resposta_texto = resposta.choices[0].message.content.strip()
    citacao = "Referências: Sabiston Textbook of Surgery 21st Edition – " + ", ".join(sorted(capitulos_usados))

    return jsonify({
        "resposta": resposta_texto,
        "referencia": citacao
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
