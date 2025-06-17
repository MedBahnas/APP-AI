import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from vector_store import VectorStore

# Charger les variables d'environnement
load_dotenv()

class Chatbot:
    """
    Classe pour gérer les conversations avec le chatbot basé sur OpenAI.
    Utilise RAG (Retrieval-Augmented Generation) pour fournir des réponses basées sur les documents.
    """
    
    def __init__(self, vector_store: VectorStore, model: str = "gpt-4-turbo"):
        """
        Initialise le chatbot.
        
        Args:
            vector_store (VectorStore): Instance du stockage vectoriel
            model (str): Modèle OpenAI à utiliser
        """
        self.vector_store = vector_store
        self.model = model
        self.conversation_history = []
        self.system_prompt = self._get_system_prompt()
        
        # Configurer l'API OpenAI
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("La clé API OpenAI n'est pas définie dans les variables d'environnement.")
        
        self.client = OpenAI(api_key=self.openai_api_key)
    
    def _get_system_prompt(self) -> str:
        """Retourne le prompt système pour guider les réponses du chatbot."""
        return """
        Tu es EduLLM, un assistant d'aide à l'apprentissage pour les étudiants en BDIA.
        Ton rôle est d'aider les étudiants à comprendre leurs cours et à réviser efficacement.
        
        Instructions importantes :
        1. Sois clair, concis et pédagogique dans tes réponses.
        2. Utilise les documents fournis comme source principale d'information.
        3. Si une question n'est pas liée aux documents, réponds de manière générale.
        4. Pour les questions complexes, fournis des explications étape par étape.
        5. Si tu ne connais pas la réponse, dis-le clairement.
        6. Utilise le format Markdown pour améliorer la lisibilité.
        """
    
    def _format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Formate les résultats de recherche en un contexte lisible.
        
        Args:
            search_results (List[Dict[str, Any]]): Résultats de la recherche vectorielle
            
        Returns:
            str: Contexte formaté pour le prompt
        """
        if not search_results:
            return "Aucune information pertinente trouvée dans les documents."
        
        context_parts = ["Informations pertinentes provenant des documents :\n"]
        
        for i, result in enumerate(search_results, 1):
            source = result['metadata'].get('title', 'Document inconnu')
            context_parts.append(f"\n--- Source {i}: {source} ---\n{result['text']}\n")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str = "") -> str:
        """
        Génère une réponse à la requête de l'utilisateur en utilisant l'API OpenAI.
        
        Args:
            query (str): Requête de l'utilisateur
            context (str): Contexte supplémentaire pour la génération
            
        Returns:
            str: Réponse générée par le modèle
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self._format_conversation_history(),
            {"role": "user", "content": query}
        ]
        
        if context:
            # Ajouter le contexte comme un message système supplémentaire
            messages.insert(1, {"role": "system", "content": f"Contexte :\n{context}"})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Erreur lors de la génération de la réponse : {str(e)}")
            return "Désolé, une erreur est survenue lors de la génération de la réponse."
    
    def _format_conversation_history(self) -> List[Dict[str, str]]:
        """
        Formate l'historique de la conversation pour l'API OpenAI.
        
        Returns:
            List[Dict[str, str]]: Historique de la conversation formaté
        """
        formatted_history = []
        
        for message in self.conversation_history[-10:]:  # Limiter à 10 derniers échanges
            role = "user" if message["is_user"] else "assistant"
            formatted_history.append({"role": role, "content": message["content"]})
        
        return formatted_history
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Traite une requête utilisateur et renvoie une réponse.
        
        Args:
            query (str): Requête de l'utilisateur
            
        Returns:
            Dict[str, Any]: Réponse et métadonnées
        """
        if not query.strip():
            return {
                "response": "Veuillez poser une question ou entrer une requête.",
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Ajouter la requête à l'historique
        self.conversation_history.append({
            "content": query,
            "is_user": True,
            "timestamp": datetime.now().isoformat()
        })
        
        # Rechercher des documents pertinents
        search_results = self.vector_store.search(query, k=3)
        context = self._format_context(search_results)
        
        # Générer une réponse
        response = self._generate_response(query, context)
        
        # Ajouter la réponse à l'historique
        self.conversation_history.append({
            "content": response,
            "is_user": False,
            "timestamp": datetime.now().isoformat()
        })
        
        # Préparer les sources pour la réponse
        sources = []
        if search_results:
            for result in search_results:
                source = result['metadata'].get('title', 'Document inconnu')
                if source not in sources:
                    sources.append(source)
        
        return {
            "response": response,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_history(self):
        """Réinitialise l'historique de la conversation."""
        self.conversation_history = []
    
    def save_conversation(self, file_path: str):
        """
        Sauvegarde l'historique de la conversation dans un fichier.
        
        Args:
            file_path (str): Chemin du fichier de sauvegarde
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
    
    def load_conversation(self, file_path: str) -> bool:
        """
        Charge un historique de conversation depuis un fichier.
        
        Args:
            file_path (str): Chemin du fichier à charger
            
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            return True
        except Exception as e:
            print(f"Erreur lors du chargement de la conversation : {str(e)}")
            return False


class QuizGenerator:
    """
    Classe pour générer des quiz à partir du contenu des documents.
    """
    
    def __init__(self, vector_store: VectorStore, model: str = "gpt-4-turbo"):
        """
        Initialise le générateur de quiz.
        
        Args:
            vector_store (VectorStore): Instance du stockage vectoriel
            model (str): Modèle OpenAI à utiliser
        """
        self.vector_store = vector_store
        self.model = model
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("La clé API OpenAI n'est pas définie dans les variables d'environnement.")
        
        self.client = OpenAI(api_key=self.openai_api_key)
    
    def generate_quiz(
        self, 
        topic: str = None, 
        num_questions: int = 5, 
        difficulty: str = "moyen",
        doc_id: str = None
    ) -> Dict[str, Any]:
        """
        Génère un quiz sur un thème donné ou à partir d'un document spécifique.
        
        Args:
            topic (str): Thème du quiz (optionnel si doc_id est fourni)
            num_questions (int): Nombre de questions à générer
            difficulty (str): Niveau de difficulté (facile, moyen, difficile)
            doc_id (str): ID du document sur lequel baser le quiz (optionnel)
            
        Returns:
            Dict[str, Any]: Quiz généré avec questions et réponses
        """
        print(f"\n=== DÉBUT GÉNÉRATION QUIZ ===")
        print(f"Topic: {topic}")
        print(f"Nombre de questions: {num_questions}")
        print(f"Difficulté: {difficulty}")
        print(f"ID Document: {doc_id}")
        
        # Vérifier qu'au moins un thème ou un document est fourni
        if not topic and not doc_id:
            return {
                "error": "Veuillez spécifier un thème ou sélectionner un document pour générer un quiz.",
                "details": "Aucun thème ou document fourni"
            }
            
        # Récupérer le contenu pertinent si un document est spécifié
        context = ""
        if doc_id:
            try:
                print(f"\n[DEBUG] Récupération des chunks pour le document {doc_id}...")
                chunks = self.vector_store.get_document_chunks(doc_id)
                if not chunks:
                    return {
                        "error": f"Aucun contenu trouvé pour le document avec l'ID {doc_id}",
                        "details": "Document vide ou introuvable"
                    }
                context = "\n\n".join([chunk.get('text', '') for chunk in chunks])
                print(f"[DEBUG] Contexte extrait (premiers 200 caractères): {context[:200]}...")
            except Exception as e:
                print(f"[ERREUR] Erreur lors de la récupération du document: {str(e)}")
                return {
                    "error": "Erreur lors de la récupération du document",
                    "details": str(e)
                }
        
        # Créer le prompt pour la génération du quiz
        try:
            print("\n[DEBUG] Création du prompt...")
            prompt = self._create_quiz_prompt(topic, num_questions, difficulty, context)
            print(f"[DEBUG] Prompt créé (premiers 200 caractères): {prompt[:200]}...")
        except Exception as e:
            print(f"[ERREUR] Erreur lors de la création du prompt: {str(e)}")
            return {
                "error": "Erreur lors de la création du prompt",
                "details": str(e)
            }
        
        # Appeler l'API pour générer le quiz
        print("\n[DEBUG] Appel à l'API OpenAI...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            print("[DEBUG] Réponse reçue de l'API")
            
            # Extraire la réponse et nettoyer les éventuels marqueurs de code
            response_content = response.choices[0].message.content
            
            # Afficher la réponse brute pour le débogage
            print("\n=== RÉPONSE BRUTE DE L'API ===")
            print(response_content)
            print("=== FIN DE LA RÉPONSE BRUTE ===")
            
            # Nettoyer la réponse pour ne garder que le contenu JSON
            original_content = response_content
            if '```json' in response_content:
                print("\n[DEBUG] Détection de ```json dans la réponse")
                parts = response_content.split('```json')
                if len(parts) > 1:
                    response_content = parts[1].split('```')[0].strip()
                    print("[DEBUG] Contenu JSON extrait après nettoyage:", response_content[:200] + "...")
            elif '```' in response_content:
                print("\n[DEBUG] Détection de ``` dans la réponse")
                parts = response_content.split('```')
                if len(parts) > 1:
                    response_content = parts[1].strip()
                    if response_content.startswith('json'):
                        response_content = response_content[4:].strip()
                    print("[DEBUG] Contenu JSON extrait après nettoyage:", response_content[:200] + "...")
            
            # Si le contenu commence par une balise XML ou HTML, essayer de l'extraire
            if '<' in response_content and '>' in response_content:
                print("\n[WARNING] Détection possible de balises HTML/XML dans la réponse")
                # Essayer d'extraire le JSON entre balises
                import re
                json_match = re.search(r'<json>([\s\S]*?)</json>', response_content, re.IGNORECASE)
                if json_match:
                    response_content = json_match.group(1).strip()
                    print("[DEBUG] JSON extrait des balises <json>:", response_content[:200] + "...")
            
            # Parser la réponse en JSON avec gestion d'erreur améliorée
            try:
                # Essayer de parser la réponse complète
                quiz_data = json.loads(response_content)
                
                # Si on a un champ 'quiz' dans la réponse, l'utiliser
                if 'quiz' in quiz_data and isinstance(quiz_data['quiz'], dict):
                    quiz_data = quiz_data['quiz']
                
                # Valider la structure de base du quiz
                if not isinstance(quiz_data, dict):
                    raise ValueError("La réponse n'est pas un objet JSON valide")
                    
                # Vérifier les champs obligatoires avec des valeurs par défaut
                quiz_data.setdefault('title', 'Quiz Généré')
                quiz_data.setdefault('description', 'Quiz généré automatiquement')
                quiz_data.setdefault('difficulty', 'moyen')
                
                # Vérifier et formater les questions
                if 'questions' not in quiz_data or not isinstance(quiz_data['questions'], list):
                    raise ValueError("Aucune question trouvée dans la réponse")
                
                # Valider chaque question
                for i, question in enumerate(quiz_data['questions'], 1):
                    if not isinstance(question, dict):
                        raise ValueError(f"La question {i} n'est pas un objet valide")
                    
                    # Définir des valeurs par défaut pour les champs manquants
                    question.setdefault('question', f"Question {i}")
                    question.setdefault('options', {"a": "Option A", "b": "Option B", "c": "Option C", "d": "Option D"})
                    question.setdefault('correct_answer', 'a')
                    question.setdefault('explanation', 'Aucune explication disponible.')
                    
                    # S'assurer que la réponse correcte est valide
                    if question['correct_answer'] not in question['options']:
                        question['correct_answer'] = next(iter(question['options'].keys()), 'a')
                        
            except json.JSONDecodeError as json_err:
                # Si le parsing échoue, essayer d'extraire le JSON de la réponse
                import re
                json_match = re.search(r'({.*})', response_content, re.DOTALL)
                if json_match:
                    try:
                        quiz_data = json.loads(json_match.group(1))
                    except:
                        raise ValueError(f"Impossible de parser la réponse JSON: {str(json_err)}")
                else:
                    raise ValueError(f"Réponse JSON invalide: {str(json_err)}")
            
            # Vérifier que les données essentielles sont présentes
            if not quiz_data or 'questions' not in quiz_data:
                return {
                    "error": "Format de quiz invalide reçu de l'API",
                    "details": f"Réponse reçue: {response_content[:200]}..."
                }
                
            return quiz_data
            
        except json.JSONDecodeError as e:
            print(f"Erreur de décodage JSON: {str(e)}")
            print(f"Réponse brute de l'API: {response_content[:500]}...")
            return {
                "error": "Erreur lors de l'analyse de la réponse du quiz.",
                "details": f"Erreur JSON: {str(e)}\nRéponse: {response_content[:200]}..."
            }
            
        except Exception as e:
            error_msg = f"Erreur lors de la génération du quiz: {str(e)}"
            print(error_msg)
            return {
                "error": "Une erreur est survenue lors de la génération du quiz.",
                "details": str(e)
            }
    
    def _create_quiz_prompt(
        self, 
        topic: str, 
        num_questions: int, 
        difficulty: str,
        context: str = ""
    ) -> str:
        """
        Crée le prompt pour la génération du quiz.
        
        Args:
            topic (str): Thème du quiz
            num_questions (int): Nombre de questions
            difficulty (str): Niveau de difficulté
            context (str): Contexte supplémentaire
            
        Returns:
            str: Prompt formaté pour l'API
        """
        # Utilisation d'une f-string pour le formatage
        prompt = f"""
        Tu es un expert en création de quiz éducatifs. Ton rôle est de créer un quiz de haute qualité
        basé sur le sujet et le niveau de difficulté fournis.
        
        Instructions :
        1. Crée un quiz avec {num_questions} questions à choix multiples.
        2. Le niveau de difficulté doit être : {difficulty}.
        3. Pour chaque question, fournis 4 options de réponse (a, b, c, d) et indique la bonne réponse.
        4. Inclus des explications claires pour chaque réponse.
        5. Le format de sortie doit être un JSON valide avec la structure suivante :
        
        {{
            "title": "Titre du quiz",
            "description": "Description du quiz",
            "difficulty": "{difficulty}",
            "questions": [
                {{
                    "question": "Texte de la question",
                    "options": {{
                        "a": "Option A",
                        "b": "Option B",
                        "c": "Option C",
                        "d": "Option D"
                    }},
                    "correct_answer": "a",
                    "explanation": "Explication détaillée de la réponse"
                }}
            ]
        }}
        """
        
        # Ajout du contexte ou du sujet
        if context:
            prompt += f"\n\nContexte du quiz :\n{context}"
        else:
            prompt += f"\n\nSujet du quiz : {topic}"
        
        return prompt
