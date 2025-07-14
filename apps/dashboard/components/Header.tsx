import React from "react";

interface HeaderProps {
  title?: string;
  subtitle?: string;
}

const Header: React.FC<HeaderProps> = ({ title = "Sentenial X Dashboard", subtitle }) => {
  return (
    <header className="bg-gray-900 text-white p-4 shadow-md">
      <div className="container mx-auto flex flex-col sm:flex-row items-center justify-between">
        <h1 className="text-2xl font-bold">{title}</h1>
        {subtitle && <p className="text-gray-400 mt-1 sm:mt-0">{subtitle}</p>}
      </div>
    </header>
  );
};

export default Header;
