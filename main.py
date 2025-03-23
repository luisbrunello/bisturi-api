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

# 📚 Carrega os dados de todos os livros
def carregar_livro(nome_json, nome_faiss, livro_id):
    with open(nome_json, "r", encoding="utf-8") as f:
        referencias = json.load(f)
    index = faiss.read_index(nome_faiss)
    return {"referencias": referencias, "index": index, "id": livro_id}

sabiston = carregar_livro("referencias.json", "indice_capitulos.faiss", "Sabiston")
anatomia = carregar_livro("referencias_anatomia.json", "indice_anatomia.faiss", "Anatomia")
mattox = carregar_livro("referencias_mattox.json", "indice_mattox.faiss", "Mattox")

bases = [sabiston, anatomia, mattox]

# 🧠 Gera embedding
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

    # 🌐 Traduz a pergunta para inglês médico
    traducao = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": "Traduza para inglês médico sem explicações."},
            {"role": "user", "content": pergunta}
        ]
    ).choices[0].message.content.strip()

    embedding = gerar_embedding(traducao).reshape(1, -1)

    contexto = ""
    fontes_usadas = []

    # 🔍 Consulta os 3 índices
    for base in bases:
        _, indices = base["index"].search(embedding, 3)

        for i in indices[0]:
            trecho = base["referencias"][i]["texto"]
            capitulo = base["referencias"][i]["capitulo"]
            contexto += f"\n[{base['id']} – {capitulo}]\n{trecho}\n"
            fontes_usadas.append(f"{base['id']} – {capitulo}")

    # 📝 Prompt com orientação científica e detalhada
    prompt = f"""
Você é um assistente médico especializado em Cirurgia Geral, altamente científico e baseado em evidências.  
Responda à pergunta abaixo **usando exclusivamente** as informações fornecidas no contexto.

Se houver informações **conflitantes entre os livros**, destaque e compare as diferenças claramente,
**citado a fonte específica abaixo de cada trecho**.

Organize sua resposta em **HTML bem formatado**, com títulos, listas, parágrafos e negrito para facilitar a leitura.

Se a resposta não estiver no contexto, responda exatamente:  
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

    # 📚 Monta citação das fontes
    fontes_unicas = sorted(set(fontes_usadas))
    citacao = "Referências utilizadas: " + "; ".join(fontes_unicas)

    return jsonify({
        "resposta": resposta_texto,
        "referencia": citacao
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

