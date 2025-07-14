import React, { useState } from "react";

interface SettingsPanelProps {
  onSave?: (settings: Settings) => void;
}

interface Settings {
  darkMode: boolean;
  notificationsEnabled: boolean;
  autoUpdate: boolean;
}

const SettingsPanel: React.FC<SettingsPanelProps> = ({ onSave }) => {
  const [settings, setSettings] = useState<Settings>({
    darkMode: false,
    notificationsEnabled: true,
    autoUpdate: false,
  });

  const handleToggle = (key: keyof Settings) => {
    setSettings((prev) => {
      const updated = { ...prev, [key]: !prev[key] };
      return updated;
    });
  };

  const handleSave = () => {
    if (onSave) {
      onSave(settings);
    }
    alert("Settings saved!");
  };

  return (
    <div className="bg-gray-800 text-white p-6 rounded shadow-md max-w-md mx-auto">
      <h2 className="text-xl font-semibold mb-4">Settings</h2>

      <div className="flex items-center justify-between mb-4">
        <label htmlFor="darkMode" className="text-gray-300">
          Dark Mode
        </label>
        <input
          id="darkMode"
          type="checkbox"
          checked={settings.darkMode}
          onChange={() => handleToggle("darkMode")}
          className="form-checkbox h-5 w-5 text-blue-600"
        />
      </div>

      <div className="flex items-center justify-between mb-4">
        <label htmlFor="notificationsEnabled" className="text-gray-300">
          Enable Notifications
        </label>
        <input
          id="notificationsEnabled"
          type="checkbox"
          checked={settings.notificationsEnabled}
          onChange={() => handleToggle("notificationsEnabled")}
          className="form-checkbox h-5 w-5 text-blue-600"
        />
      </div>

      <div className="flex items-center justify-between mb-6">
        <label htmlFor="autoUpdate" className="text-gray-300">
          Auto Update
        </label>
        <input
          id="autoUpdate"
          type="checkbox"
          checked={settings.autoUpdate}
          onChange={() => handleToggle("autoUpdate")}
          className="form-checkbox h-5 w-5 text-blue-600"
        />
      </div>

      <button
        onClick={handleSave}
        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded"
      >
        Save Settings
      </button>
    </div>
  );
};

export default SettingsPanel;
