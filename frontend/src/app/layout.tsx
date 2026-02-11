import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000"),
  title: "RideWise | AI-Powered Churn Prediction Dashboard",
  description: "Predict customer churn with machine learning. Analyze rider behavior, understand risk factors with SHAP explanations, and take proactive retention actions.",
  keywords: ["churn prediction", "machine learning", "customer analytics", "SHAP", "ride sharing", "retention"],
  authors: [{ name: "RideWise Analytics" }],
  openGraph: {
    title: "RideWise Churn Prediction Dashboard",
    description: "AI-powered customer analytics for proactive retention strategies",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "RideWise Churn Prediction Dashboard",
    description: "AI-powered customer analytics for proactive retention strategies",
  },
  robots: { index: true, follow: true },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrains.variable}`}>
      <body className="font-sans min-h-screen">
        {children}
      </body>
    </html>
  );
}
