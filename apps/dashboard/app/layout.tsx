import "../styles/global.css";
import React from "react";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Sentenial X Dashboard",
  description: "Advanced threat emulation & telemetry intelligence suite",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-white min-h-screen font-sans antialiased">
        <div className="max-w-screen-xl mx-auto px-4 py-6">
          {/* Add shared components here (Header, Sidebar, etc.) */}
          {children}
        </div>
      </body>
    </html>
  );
}

