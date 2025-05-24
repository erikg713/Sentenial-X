# sentenial_core/compliance/legal_ontology_parser.py

"""
Legal Ontology Parser
---------------------
Parses and structures legal and regulatory ontologies such as GDPR, HIPAA, NIST.
Transforms natural language regulations into machine-readable semantic graphs
to enable compliance reasoning and automated impact analysis.

This module leverages NLP transformers for entity extraction, ontology graph building,
and reasoning-friendly data structures.
"""

from typing import List, Dict, Any
import spacy
import networkx as nx
import json
import logging

logger = logging.getLogger(__name__)

class LegalOntologyParser:
    def __init__(self, model_name: str = "en_core_web_trf"):
        # Load a transformer-based spaCy model for best entity & relation extraction
        self.nlp = spacy.load(model_name)
        self.graph = nx.DiGraph()

    def parse_text(self, legal_text: str) -> nx.DiGraph:
        """
        Parses legal text into an ontology graph where nodes represent legal concepts,
        obligations, rights, and constraints, and edges denote relationships.

        Args:
            legal_text (str): Raw legal text to parse.

        Returns:
            networkx.DiGraph: Directed graph representing ontology.
        """
        doc = self.nlp(legal_text)

        # Extract key entities & concepts
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Extract noun chunks as candidate concepts
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]

        logger.debug(f"Extracted entities: {entities}")
        logger.debug(f"Extracted noun chunks: {noun_chunks}")

        # Add entities as nodes
        for text, label in entities:
            self.graph.add_node(text, label=label, type="entity")

        # Add noun chunks as nodes
        for chunk in noun_chunks:
            if chunk not in self.graph.nodes:
                self.graph.add_node(chunk, type="concept")

        # Add edges based on dependency parse relations to capture semantic relations
        for token in doc:
            if token.dep_ in ("nsubj", "dobj", "pobj", "attr", "appos"):
                head_text = token.head.text
                token_text = token.text
                if head_text in self.graph.nodes and token_text in self.graph.nodes:
                    self.graph.add_edge(head_text, token_text, relation=token.dep_)

        logger.debug(f"Constructed graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")

        return self.graph

    def export_graph_json(self) -> str:
        """
        Exports the ontology graph to JSON for downstream usage.

        Returns:
            str: JSON string representing the graph.
        """
        data = nx.node_link_data(self.graph)
        json_data = json.dumps(data, indent=2)
        logger.debug("Exported ontology graph to JSON")
        return json_data

    def load_graph_from_json(self, json_data: str) -> nx.DiGraph:
        """
        Loads ontology graph from a JSON string.

        Args:
            json_data (str): JSON representation of the graph.

        Returns:
            networkx.DiGraph: Loaded graph.
        """
        data = json.loads(json_data)
        self.graph = nx.node_link_graph(data)
        logger.debug(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges from JSON")
        return self.graph

# Example Usage (for unit testing or CLI tool):
if __name__ == "__main__":
    sample_text = """
    Under GDPR, data controllers must obtain explicit consent before processing personal data.
    The data subject has the right to access and erase their data at any time.
    Failure to comply can result in penalties up to 4% of annual global turnover.
    """
    parser = LegalOntologyParser()
    ontology_graph = parser.parse_text(sample_text)
    print(parser.export_graph_json())
