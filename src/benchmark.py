import time
import json
from query import answer_question
import matplotlib.pyplot as plt

MODELS = ["mistral:7b", "qwen3.6:latest", "qwen2.5:72b"]

QUESTIONS = [
    "Quels sont les avantages des transformers ?",
    "Qu'est-ce que le mécanisme d'attention dans l'architecture Transformer ?",
    "Quels sont les points principaux abordés dans le document sur la coévolution ?",
    "Quelle méthode est utilisée pour approximer les statistiques du fond d'ondes gravitationnelles ?",
]


def run_benchmark():
    results = []

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Modèle : {model}")
        print(f"{'='*60}")

        for question in QUESTIONS:
            debut = time.time()
            try:
                reponse, sources = answer_question(question, model=model)
                duree = time.time() - debut
                statut = "ok"
            except Exception as e:
                duree = time.time() - debut
                reponse = str(e)
                sources = []
                statut = "erreur"

            print(f"\nQuestion : {question}")
            print(f"Temps : {duree:.2f}s | Statut : {statut}")

            results.append({
                "model": model,
                "question": question,
                "duree_secondes": round(duree, 2),
                "statut": statut,
                "nb_sources": len(sources),
            })

    return results


def afficher_resume(results):
    print(f"\n\n{'='*60}")
    print("RÉSUMÉ DU BENCHMARK")
    print(f"{'='*60}\n")

    for model in MODELS:
        temps_model = [r["duree_secondes"] for r in results if r["model"] == model and r["statut"] == "ok"]
        if temps_model:
            moyenne = sum(temps_model) / len(temps_model)
            print(f"{model} : temps moyen = {moyenne:.2f}s (sur {len(temps_model)} questions réussies)")
        else:
            print(f"{model} : aucune réponse réussie")


def sauvegarder_resultats(results, chemin="benchmark_results.json"):
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nRésultats sauvegardés dans {chemin}")

def generer_graphique(results, chemin="benchmark_chart.png"):
    questions = sorted(set(r["question"] for r in results), key=lambda q: QUESTIONS.index(q))
    x = range(len(questions))
    largeur = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(MODELS):
        temps = [
            next(r["duree_secondes"] for r in results if r["model"] == model and r["question"] == q)
            for q in questions
        ]
        positions = [pos + i * largeur for pos in x]
        ax.bar(positions, temps, width=largeur, label=model)

    ax.set_xlabel("Question")
    ax.set_ylabel("Temps (secondes)")
    ax.set_title("Temps de réponse par modèle et par question")
    ax.set_xticks([pos + largeur for pos in x])
    ax.set_xticklabels([f"Q{i+1}" for i in range(len(questions))])
    ax.legend()
    plt.tight_layout()
    plt.savefig(chemin)
    print(f"\nGraphique sauvegardé dans {chemin}")

if __name__ == "__main__":
    results = run_benchmark()
    afficher_resume(results)
    sauvegarder_resultats(results)
    generer_graphique(results)