 @app.get("/healthz")
 def health():
-    return {"status": "ok"}
+    return {
+        "status": "ok",
+        "name": "Sentenial-X A.I.",
+        "tagline": "Crafted for resilience. Engineered for vengeance."
+    }
