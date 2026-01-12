import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Toaster } from "sonner";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Statement Extractor | corp-o-rate",
  description: "Extract structured statements from text using our T5-Gemma 2 model. Identify subjects, objects, and predicates with entity type recognition.",
  keywords: ["NLP", "statement extraction", "named entity recognition", "T5", "Gemma", "machine learning", "corp-o-rate"],
  authors: [{ name: "corp-o-rate" }],
  openGraph: {
    title: "Statement Extractor | corp-o-rate",
    description: "Extract structured statements from text using our T5-Gemma 2 model.",
    url: "https://extractor.corp-o-rate.com",
    siteName: "Statement Extractor",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Statement Extractor | corp-o-rate",
    description: "Extract structured statements from text using our T5-Gemma 2 model.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen`}
      >
        {children}
        <Toaster
          position="top-center"
          expand={false}
          richColors
          closeButton
        />
      </body>
    </html>
  );
}
