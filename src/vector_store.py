import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
from document_processor import DocumentProcessor

# Charger les variables d'environnement
load_dotenv()

class VectorStore:
    """
    Classe pour gérer le stockage et la recherche de documents vectorisés.
    Utilise FAISS pour la recherche de similarité et OpenAI pour les embeddings.
    """
    
    def __init__(self, index_dir: str = "data/indices", model_name: str = "text-embedding-3-small"):
        """
        Initialise le stockage vectoriel.
        
        Args:
            index_dir (str): Répertoire pour stocker les indices FAISS
            model_name (str): Nom du modèle d'embedding à utiliser
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.dimension = 1536  # Dimension par défaut pour text-embedding-3-small
        self.index = None
        self.documents = []
        self.doc_processor = DocumentProcessor()
        
        # Initialiser OpenAI
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("La clé API OpenAI n'est pas définie dans les variables d'environnement.")
        
        openai.api_key = self.openai_api_key
        
        # Initialiser le modèle local pour les opérations de texte léger
        self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Génère un embedding pour le texte donné en utilisant l'API OpenAI.
        
        Args:
            text (str): Texte à vectoriser
            
        Returns:
            np.ndarray: Vecteur d'embedding
        """
        try:
            response = openai.Embedding.create(
                input=text,
                model=self.model_name
            )
            return np.array(response['data'][0]['embedding'], dtype='float32')
        except Exception as e:
            print(f"Erreur lors de la génération de l'embedding: {str(e)}")
            # En cas d'erreur, utiliser un modèle local comme solution de secours
            return self.local_model.encode(text, convert_to_numpy=True).astype('float32')
    
    def add_document(self, file_path: str) -> str:
        """
        Ajoute un document à la base de données vectorielle.
        
        Args:
            file_path (str): Chemin vers le fichier à ajouter
            
        Returns:
            str: ID du document ajouté
        """
        # Traiter le document
        result = self.doc_processor.process_document(file_path)
        
        # Créer un ID unique pour le document
        doc_id = f"doc_{len(self.documents) + 1}"
        
        # Découper le contenu en morceaux
        chunks = self.doc_processor.split_into_chunks(result['content'])
        
        # Ajouter les métadonnées à chaque morceau
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i+1}"
            self.documents.append({
                'id': chunk_id,
                'doc_id': doc_id,
                'file_path': file_path,
                'text': chunk['text'],
                'start': chunk['start'],
                'end': chunk['end'],
                'metadata': {
                    'title': result['metadata'].get('title', os.path.basename(file_path)),
                    'type': result['metadata'].get('type', 'unknown'),
                    'source': file_path
                }
            })
        
        # Mettre à jour l'index avec les nouveaux vecteurs
        self._update_index()
        
        return doc_id
    
    def _update_index(self):
        """Met à jour l'index FAISS avec les documents actuels."""
        if not self.documents:
            return
        
        # Générer les embeddings pour tous les documents
        texts = [doc['text'] for doc in self.documents]
        embeddings = np.array([self._get_embedding(text) for text in texts], dtype='float32')
        
        # Créer ou mettre à jour l'index FAISS
        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Vérifier si l'index doit être redimensionné
        if self.index.ntotal > 0:
            self.index.add(embeddings)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Effectue une recherche sémantique dans les documents.
        
        Args:
            query (str): Requête de recherche
            k (int): Nombre de résultats à retourner
            
        Returns:
            List[Dict[str, Any]]: Liste des résultats de recherche avec scores
        """
        if not self.documents or self.index is None:
            return []
        
        # Générer l'embedding pour la requête
        query_embedding = self._get_embedding(query)
        query_embedding = np.array([query_embedding], dtype='float32')
        
        # Rechercher les k plus proches voisins
        distances, indices = self.index.search(query_embedding, k)
        
        # Récupérer les documents correspondants
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Vérifier les limites
                doc = self.documents[idx]
                results.append({
                    'id': doc['id'],
                    'doc_id': doc['doc_id'],
                    'text': doc['text'],
                    'score': float(distances[0][i]),
                    'metadata': doc['metadata'],
                    'start': doc['start'],
                    'end': doc['end']
                })
        
        return results
    
    def save_index(self, index_name: str = "default"):
        """
        Sauvegarde l'index et les métadonnées sur le disque.
        
        Args:
            index_name (str): Nom du fichier d'index
        """
        if self.index is None:
            return
        
        # Sauvegarder l'index FAISS
        index_path = self.index_dir / f"{index_name}.index"
        faiss.write_index(self.index, str(index_path))
        
        # Sauvegarder les métadonnées des documents
        metadata_path = self.index_dir / f"{index_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def load_index(self, index_name: str = "default") -> bool:
        """
        Charge un index et ses métadonnées depuis le disque.
        
        Args:
            index_name (str): Nom du fichier d'index
            
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        index_path = self.index_dir / f"{index_name}.index"
        metadata_path = self.index_dir / f"{index_name}_metadata.json"
        
        if not index_path.exists() or not metadata_path.exists():
            return False
        
        try:
            # Charger l'index FAISS
            self.index = faiss.read_index(str(index_path))
            
            # Charger les métadonnées des documents
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            return True
        except Exception as e:
            print(f"Erreur lors du chargement de l'index: {str(e)}")
            return False
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Récupère tous les morceaux d'un document spécifique.
        
        Args:
            doc_id (str): ID du document
            
        Returns:
            List[Dict[str, Any]]: Liste des morceaux du document
        """
        return [chunk for chunk in self.documents if chunk['doc_id'] == doc_id]
