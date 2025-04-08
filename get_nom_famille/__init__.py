import azure.functions as func
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_ner_model():
    logger.info("Chargement du modèle NER...")
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
    return pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

nlp = load_ner_model()

class InformationExtractor:
    def __init__(self , nlp_pipeline):
        self.nlp=nlp_pipeline
        logger.info("Modèle NER initialisé avec succès.")
    def check_noun(self, msg_2_check):
        logger.debug(f"Vérification du nom : {msg_2_check}")
        def check_str(msg_2_check: str) -> bool:
            return isinstance(msg_2_check, str) and bool(msg_2_check.strip()) and any(ele in msg_2_check for ele in ["a", "e", "i", "o", "u", "y"])
        if not check_str(msg_2_check):
            logger.warning(f"Le message {msg_2_check} n'est pas une chaîne valide.")
            return False
        if not re.match(r"^[a-zA-ZÀ-ÿ' -]+$", msg_2_check):
            logger.warning(f"Le message {msg_2_check} contient des caractères invalides.")
            return False
        return True
    def extraire_nom(self, texte):
        logger.info(f"Extraction du nom à partir du texte : {texte}")
        entities = self.nlp(texte)
        for ent in entities:
            if ent['entity_group'] == "PER":
                if self.check_noun(ent['word'].lower()):
                    logger.info(f"Nom extrait : {ent['word'].upper()}")
                    return ent['word'].upper()
        logger.warning("Aucun nom n'a été extrait.")
        return None
      
    
extractor = InformationExtractor(nlp)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
        query = req_body.get('text')

        if not query:
            return func.HttpResponse(
                json.dumps({"error": "No query provided in request body"}),
                mimetype="application/json",
                status_code=400
            )

        result = extractor.extraire_nom(query)

        return func.HttpResponse(
            json.dumps({"response": result}),
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
