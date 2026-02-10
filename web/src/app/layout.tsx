import type { Metadata } from "next";
import { Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-outfit",
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL("https://ridewise.vercel.app"),
  title: "RideWise | AI-Powered Customer Analytics for Mobility",
  description:
    "Transform your ride-sharing business with intelligent customer analytics, churn prediction, and real-time insights. Reduce churn by up to 40% with data-driven retention strategies.",
  keywords: [
    "customer analytics",
    "churn prediction",
    "ride sharing",
    "mobility tech",
    "machine learning",
    "retention strategies",
  ],
  authors: [{ name: "RideWise" }],
  openGraph: {
    title: "RideWise | AI-Powered Customer Analytics for Mobility",
    description:
      "Transform your ride-sharing business with intelligent customer analytics, churn prediction, and real-time insights.",
    url: "https://ridewise.vercel.app",
    siteName: "RideWise",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "RideWise - Customer Analytics Platform",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "RideWise | AI-Powered Customer Analytics for Mobility",
    description:
      "Transform your ride-sharing business with intelligent customer analytics and churn prediction.",
    images: ["/og-image.png"],
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={outfit.variable}>
      <body className="font-sans">{children}</body>
    </html>
  );
}
