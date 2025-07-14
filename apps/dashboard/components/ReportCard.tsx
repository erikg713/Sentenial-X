import React from "react";

interface ReportCardProps {
  title: string;
  value: string | number;
  description?: string;
  color?: "blue" | "green" | "red" | "yellow" | "gray";
}

const colorClasses: Record<string, string> = {
  blue: "bg-blue-500 text-white",
  green: "bg-green-500 text-white",
  red: "bg-red-500 text-white",
  yellow: "bg-yellow-400 text-black",
  gray: "bg-gray-300 text-black",
};

const ReportCard: React.FC<ReportCardProps> = ({
  title,
  value,
  description,
  color = "gray",
}) => {
  const bgClass = colorClasses[color] || colorClasses.gray;

  return (
    <div className={`rounded shadow-md p-4 ${bgClass} max-w-xs`}>
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className="text-3xl font-bold">{value}</p>
      {description && <p className="mt-1 text-sm opacity-80">{description}</p>}
    </div>
  );
};

export default ReportCard;

