import React, { useState } from "react";

interface SimulationControlsProps {
  onStart?: () => void;
  onPause?: () => void;
  onReset?: () => void;
  isRunning?: boolean;
}

const SimulationControls: React.FC<SimulationControlsProps> = ({
  onStart,
  onPause,
  onReset,
  isRunning = false,
}) => {
  const [running, setRunning] = useState(isRunning);

  const handleStart = () => {
    setRunning(true);
    onStart && onStart();
  };

  const handlePause = () => {
    setRunning(false);
    onPause && onPause();
  };

  const handleReset = () => {
    setRunning(false);
    onReset && onReset();
  };

  return (
    <div className="flex gap-4 justify-center mt-4">
      {!running ? (
        <button
          onClick={handleStart}
          className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded"
        >
          Start
        </button>
      ) : (
        <button
          onClick={handlePause}
          className="bg-yellow-500 hover:bg-yellow-600 text-white font-semibold py-2 px-4 rounded"
        >
          Pause
        </button>
      )}

      <button
        onClick={handleReset}
        className="bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded"
      >
        Reset
      </button>
    </div>
  );
};

export default SimulationControls;


