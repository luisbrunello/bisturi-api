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

# Carrega as referências e índices dos 3 livros
livros = [
    {
        "nome": "Sabiston Textbook of Surgery 21st Ed",
        "referencias": json.load(open("referencias.json", "r", encoding="utf-8")),
        "index": faiss.read_index("indice_capitulos.faiss")
    },
    {
        "nome": "Surgical Anatomy and Technique – Skandalakis 5th Ed",
        "referencias": json.load(open("referencias_anatomia.json", "r", encoding="utf-8")),
        "index": faiss.read_index("indice_anatomia.faiss")
    },
    {
        "nome": "Mattox Trauma 9th Ed",
        "referencias": json.load(open("referencias_mattox.json", "r", encoding="utf-8")),
        "index": faiss.read_index("indice_mattox.faiss")
    }
]

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

    contexto = ""
    referencias_usadas = []

    for livro in livros:
        _, indices = livro["index"].search(embedding, 3)
        for i in indices[0]:
            ref = livro["referencias"][i]
            capitulo = ref.get("capitulo", "Capítulo desconhecido")
            trecho = ref.get("texto", "")
            contexto += f"\n[{livro['nome']} – {capitulo}]\n{trecho}\n"
            referencias_usadas.append(f"{livro['nome']} – {capitulo}")

    # Prompt final
    prompt = f"""
Você é um assistente médico especializado em Cirurgia Geral e altamente científico.  
Responda à pergunta abaixo usando somente as informações contidas no contexto fornecido.  
Não use conhecimento próprio e não adicione dados externos, mesmo que saiba a resposta.
Caso diferentes livros tragam informações divergentes, explique as diferenças com base no material e cite as fontes diretamente abaixo de cada ponto.
Organize a resposta em HTML, com títulos, listas e parágrafos para facilitar a leitura.

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

    referencias_formatadas = sorted(set(referencias_usadas))
    citacoes_html = "<br>".join(referencias_formatadas)

    return jsonify({
        "resposta": resposta_texto,
        "referencia": f"<b>Referências:</b><br>{citacoes_html}"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
