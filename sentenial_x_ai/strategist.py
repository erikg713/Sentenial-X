from learner import learn_from_memory

def strategize():
    stats = learn_from_memory()
    if stats.get("shutdown", 0) >= 3:
        print("[STRATEGY] Too many shutdown attempts. Activating lockdown.")
        # Add defensive logic here
