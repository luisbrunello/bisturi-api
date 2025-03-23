from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import faiss
import json
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Carrega os dados do livro
with open("referencias.json", "r", encoding="utf-8") as f:
    referencias = json.load(f)

index = faiss.read_index("indice_capitulos.faiss")

# Gera embedding
def gerar_embedding(texto):
    response = client.embeddings.create(
        input=texto,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding)

@app.route("/perguntar", methods=["POST"])
def perguntar():
    dados = request.get_json()
    pergunta = dados.get("pergunta")

    if not pergunta:
        return jsonify({"erro": "Pergunta não fornecida."}), 400

    # Traduz a pergunta para inglês médico antes do embedding
    traducao = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": "Traduza para inglês médico sem explicações."},
            {"role": "user", "content": pergunta}
        ]
    ).choices[0].message.content.strip()

    embedding = gerar_embedding(traducao).reshape(1, -1)
    _, indices = index.search(embedding, 3)

    contexto = ""
    capitulos_usados = set()

    for i in indices[0]:
        trecho = referencias[i]["texto"]
        capitulo = referencias[i]["capitulo"]
        contexto += f"\n[{capitulo}]\n{trecho}\n"
        capitulos_usados.add(capitulo)

    # Prompt final
    prompt = f"""
Você é um assistente médico especializado em Cirurgia Geral e altamente científico.  
Responda à pergunta abaixo usando somente as informações contidas no contexto fornecido.  
Não use conhecimento próprio e não adicione dados externos, mesmo que saiba a resposta.
Apresente a resposta da maneira mais completa possível.
Organize a resposta em HTML, com títulos, listas e parágrafos, para facilitar a leitura.
Caso a informação não esteja no contexto, responda exatamente:  
<b>Essa informação não está disponível no material fornecido.</b>

---

<h3>Contexto:</h3>
<pre>{contexto}</pre>

<h3>Pergunta:</h3>
<p>{pergunta}</p>
"""

    resposta = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": prompt}
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
