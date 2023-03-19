# GPT-3 Swarm

Ce projet fournit une implémentation en Python pour exécuter plusieurs modèles GPT-3 de OpenAI en parallèle, ce qui est particulièrement utile pour les applications à haut débit. L'implémentation utilise asyncio et ZeroMQ pour communiquer entre les travailleurs et le collecteur, et gère les restrictions de taux imposées par OpenAI. Le code fournit une séparation claire des préoccupations, la logique de l'essaim et le client GPT-3 étant dans des classes différentes. Le projet comprend également un client d'exemple qui montre comment utiliser l'implémentation pour générer des réponses à plusieurs messages simultanément.

## Caractéristiques

- Prend en charge l'exécution simultanée de plusieurs modèles GPT-3.
- Utilise asyncio et ZeroMQ pour une communication efficace entre les travailleurs et le collecteur.
- Gère les restrictions de taux imposées par OpenAI.
- Fournit une séparation claire des préoccupations entre la logique de l'essaim et le client GPT-3.
- Comprend un client d'exemple qui montre comment utiliser l'implémentation.

## Technologies utilisées

- Python 3.8
- OpenAI API
- asyncio
- ZeroMQ

## Installation

- Clonez le dépôt.
- Installez les dépendances à l'aide de pip.
- Configurez la clé API OpenAI dans le fichier de configuration.
- Exécutez le client d'exemple pour générer des réponses à plusieurs messages simultanément.

## 
```bash
    python -m venv env 
    source env/bin/activate 
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXX; python main.py start-swarming
```

## Contribuer

Les contributions sont les bienvenues ! Veuillez ouvrir un problème ou une demande de tirage si vous avez des suggestions ou des améliorations.

## Licence

Ce projet est sous licence MIT.
