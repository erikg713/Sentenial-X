# tests/test_compliance_module.py

import unittest
from sentenial_core.compliance.legal_ontology_parser import LegalOntologyParser
from sentenial_core.compliance.regulatory_vector_matcher import RegulatoryVectorMatcher
from sentenial_core.compliance.ai_audit_tracer import AIAuditTracer
from sentenial_core.compliance.fine_impact_estimator import FineImpactEstimator

class TestComplianceModules(unittest.TestCase):
    def setUp(self):
        self.parser = LegalOntologyParser()
        self.matcher = RegulatoryVectorMatcher()
        self.tracer = AIAuditTracer(audit_log_file="test_audit.log")
        self.estimator = FineImpactEstimator()

    def test_legal_ontology_parser(self):
        sample_text = "GDPR requires data protection by design and default."
        graph = self.parser.parse_text(sample_text)
        self.assertGreater(graph.number_of_nodes(), 0)
        self.assertGreater(graph.number_of_edges(), 0)

    def test_regulatory_vector_matcher(self):
        regs = ["Personal data must be encrypted."]
        matches = self.matcher.match_regulations_to_controls(regs, threshold=0.5)
        self.assertIn(regs[0], matches)
        self.assertTrue(len(matches[regs[0]]) > 0)

    def test_ai_audit_tracer(self):
        decision = {"action": "test"}
        self.tracer.log_decision(decision)
        logs = self.tracer.load_audit_log()
        self.assertTrue(any(log["decision"] == decision for log in logs))

    def test_fine_impact_estimator(self):
        fine = self.estimator.estimate_fine("GDPR", 2, "medium")
        self.assertGreater(fine, 0)

if __name__ == "__main__":
    unittest.main()
