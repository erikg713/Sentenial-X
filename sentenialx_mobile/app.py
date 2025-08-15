from sentenialx_mobile.services.models_updater import start_models_updater

def on_start(self):
    ...
    self._model_updater_stop = start_models_updater(lambda: self.state.update_interval_seconds)
    bus.on("model:update", lambda p: self.refresh_ui())
    bus.on("toast", lambda msg: Snackbar(text=str(msg)).open())
