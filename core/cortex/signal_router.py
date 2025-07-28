class SignalRouter:
    def __init__(self, brainstem, analyzer, engine):
        self.brainstem = brainstem
        self.analyzer = analyzer
        self.engine = engine

    def handle(self, signal):
        # Step 1: Brainstem reflex
        reflex = self.brainstem.process_signal(signal)

        # Step 2: Semantic interpretation
        semantic_result = self.analyzer.analyze(signal)

        # Step 3: Strategic decision
        decision = self.engine.evaluate(signal, semantic_result)

        return {
            "reflex": reflex,
            "semantic": semantic_result,
            "decision": decision
        }
